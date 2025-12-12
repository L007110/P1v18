import numpy as np
import collections
from ChannelModel import global_channel_model
from logger import debug
import Parameters  # 动态读取 ABLATION_MODE
from Parameters import (
    V2V_DELAY_THRESHOLD, V2I_LINK_POSITIONS, TRANSMITTDE_POWER,
    V2V_CHANNEL_BANDWIDTH, V2V_PACKET_SIZE_BITS, V2V_MIN_SNR_DB,
    REWARD_RUNNING_WINDOW
)


class NewRewardCalculator:
    def __init__(self):
        self.channel_model = global_channel_model
        self.BEAM_ROLLOFF_EXPONENT = 2
        self.ANGLE_PER_DIRECTION = 10

        # 保持 P1v17 成功的归一化设置
        window = getattr(Parameters, 'REWARD_RUNNING_WINDOW', 2000)
        self.stats = {
            'snr': {'buf': collections.deque(maxlen=window), 'min': -100.0, 'max': 100.0},
            'delay': {'buf': collections.deque(maxlen=window), 'min': 0.0, 'max': 2.0},
            'v2i': {'buf': collections.deque(maxlen=window), 'min': 1e-15, 'max': 1e-3},
            'power': {'buf': collections.deque(maxlen=window), 'min': 0.0, 'max': 1.0}
        }
        debug(f"NewRewardCalculator (P1v18 Optimized Ablation) initialized")

    def _update_stats(self, key, value):
        # 保持 P1v17 的每步更新逻辑
        s = self.stats[key]
        s['buf'].append(value)
        if value < s['min']: s['min'] = value
        if value > s['max']: s['max'] = value

    def _normalize(self, key, value, invert=False):
        self._update_stats(key, value)
        s = self.stats[key]
        denom = s['max'] - s['min']
        if abs(denom) < 1e-15: return 0.5
        clipped = max(s['min'], min(value, s['max']))
        norm = (clipped - s['min']) / denom
        return 1.0 - norm if invert else norm

    def _calculate_directional_gain(self, h_dir, v_dir):
        theta_h = (h_dir - 1) * 10
        theta_v = (1 - v_dir) * 10
        gain = (np.cos(np.deg2rad(theta_h)) * np.cos(np.deg2rad(theta_v))) ** 2
        return gain

    def calculate_delay(self, dist, action, gain, snr_lin):
        try:
            prop = dist / 3e8
            if snr_lin is None or snr_lin <= 0: return 1.0
            rate = V2V_CHANNEL_BANDWIDTH * np.log2(1 + snr_lin)
            trans = V2V_PACKET_SIZE_BITS / (rate + 1e-9)
            return prop + trans
        except:
            return 1.0

    def _record_communication_metrics(self, dqn, delay, snr):
        dqn.delay_list.append(delay)
        dqn.snr_list.append(snr)
        succ = 1 if (delay <= V2V_DELAY_THRESHOLD and snr >= V2V_MIN_SNR_DB) else 0
        if not hasattr(dqn, 'v2v_success_list'): dqn.v2v_success_list = []
        dqn.v2v_success_list.append(succ)

    # 关键：保留此函数防止报错
    def get_csi_for_state(self, vehicle, dqn):
        if vehicle is None: return [0.0] * 5
        try:
            csi = self.channel_model.get_channel_state_info(
                dqn.bs_loc, vehicle.curr_loc,
                tx_power=TRANSMITTDE_POWER, bandwidth=V2V_CHANNEL_BANDWIDTH
            )
            return [csi['distance_3d'], csi['path_loss_total_db'],
                    csi['shadowing_db'], csi['snr_db'], getattr(dqn, 'prev_snr', 0.0)]
        except:
            return [0.0] * 5

    def calculate_complete_reward(self, dqn, vehicles, action, active_interferers=None):
        if not vehicles:
            self._record_communication_metrics(dqn, 1.0, -100.0)
            return 0.0, {}

        try:
            # 1. 物理计算
            vehicle = vehicles[0]
            dist = self.channel_model.calculate_3d_distance(dqn.bs_loc, vehicle.curr_loc)

            beam_count = action[0] + 1
            power_ratio = (action[3] + 1) / 10.0
            gain = self._calculate_directional_gain(action[1], action[2])
            total_power = TRANSMITTDE_POWER * power_ratio * beam_count * gain

            # V2V 干扰
            interf_W = 0.0
            if active_interferers:
                for interf in active_interferers:
                    if interf['tx_pos'] == dqn.bs_loc: continue
                    d = self.channel_model.calculate_3d_distance(interf['tx_pos'], vehicle.curr_loc)
                    pl, _, _ = self.channel_model.calculate_path_loss(d)
                    interf_W += interf['power_W'] * (10 ** (-pl / 10))

            # SINR
            pl, _, _ = self.channel_model.calculate_path_loss(dist)
            sig_W = total_power * (10 ** (-pl / 10))
            noise = self.channel_model._calculate_noise_power(V2V_CHANNEL_BANDWIDTH)
            sinr_lin = sig_W / (interf_W + noise + 1e-20)
            snr_db = 10 * np.log10(max(sinr_lin, 1e-20))
            delay = self.calculate_delay(dist, action, gain, sinr_lin)

            # V2I 干扰
            v2i_W = 0.0
            for link in V2I_LINK_POSITIONS:
                d = self.channel_model.calculate_3d_distance(dqn.bs_loc, link['rx'])
                pl, _, _ = self.channel_model.calculate_path_loss(d)
                v2i_W += total_power * (10 ** (-pl / 10))

            # 2. 归一化 (Normalization)
            n_snr = self._normalize('snr', snr_db, invert=False)
            n_delay = self._normalize('delay', delay, invert=True)
            n_v2i = self._normalize('v2i', v2i_W, invert=True)
            n_power = self._normalize('power', power_ratio, invert=True)

            # 3. 加权 (使用 Parameters 中的固定最优值)
            t_snr = Parameters.SNR_MULTIPLIER * n_snr
            t_delay = Parameters.DELAY_MULTIPLIER * n_delay
            t_v2i = Parameters.V2I_MULTIPLIER * n_v2i
            t_power = Parameters.POWER_MULTIPLIER * n_power

            # 4. P1v18 消融逻辑 (Ablation Logic)
            # 根据模式强行置零
            mode = getattr(Parameters, 'ABLATION_MODE', 'E0')

            if mode == "E1":  # No V2I Penalty
                t_v2i = 0.0
            elif mode == "E2":  # No Delay Reward
                t_delay = 0.0
            elif mode == "E3":  # Only SNR (Greedy)
                t_delay = 0.0
                t_v2i = 0.0
                t_power = 0.0

            # (E0 模式下所有项都保留)

            reward = t_snr + t_delay + t_v2i + t_power

            # 更新状态
            dqn.prev_v2v_interference = interf_W
            dqn.prev_snr = snr_db
            self._record_communication_metrics(dqn, delay, snr_db)

            breakdown = {
                'raw_snr': snr_db, 'norm_snr': n_snr,
                'norm_v2i': n_v2i, 'total_reward': reward
            }
            return reward, breakdown

        except Exception as e:
            debug(f"Reward Error: {e}")
            return 0.0, {}


new_reward_calculator = NewRewardCalculator()