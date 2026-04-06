import numpy as np
import numpy.typing as npt

from MRR.Evaluator import build_Evaluator
from MRR.gragh import plot
from MRR.logger import Logger
from MRR.Simulator import Ring, build_TransferFunction


class Model:
    """
    Args:
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                number_of_episodes_in_L (int): Number of the episodes in L.
                number_of_episodes_in_K (int): Number of the episodes in K.
                number_of_steps (int): Number of steps.
                center_wavelength (float): The center wavelength.
                number_of_rings (int): Number of rings. The ring order.
                H_p (float): The threshold of the max loss in pass band. loss_p.
                H_s (float): The threshold of the min loss in stop band. loss_s.
                length_of_3db_band (float): The required length of the 3dB band.
                FSR (float): The required FSR.
                alpha (float): The propagation loss coefficient.
                eta (float): The coupling loss coefficient.
                n_eff (float): The effective refractive index.
                n_g (float): The group index.
                min_ring_length (float): The minimum round-trip length.
    Attributes:
        eta (float): The coupling loss coefficient.
        number_of_episodes_in_L (int): Number of the episodes in L.
        number_of_episodes_in_K (int): Number of the episodes in K.
        number_of_steps (int): Number of steps.
        number_of_rings (int): Number of rings. The ring order.
        required_FSR (float): The required FSR.
        logger (Logger): Logger.
        ring (Ring): The minimum round-trip length.
        Evaluator (Evaluator): Evaluator.
        MRR (MRR): MRR.
    """

    def __init__(self, config):
        self.eta = config["eta"]
        self.number_of_episodes_in_L = config["number_of_episodes_in_L"]
        self.number_of_episodes_in_K = config["number_of_episodes_in_K"]
        self.number_of_steps = config["number_of_steps"]
        self.number_of_rings = config["number_of_rings"]
        self.required_FSR = config["FSR"]
        self.logger = Logger()
        self.logger.save_optimization_config(config)
        self.ring = Ring(config)
        self.Evaluator = build_Evaluator(config)
        self.TransferFunction = build_TransferFunction(config)
        self.L_list = []
        self.K_list = []
        self.Q_list = []
        self.rng = config.get_multi_start_local_search_rng()

    def init_K(self) -> npt.NDArray[np.float_]:
        return np.array([self.rng.uniform(0, self.eta) for _ in range(self.number_of_rings + 1)])

    def optimize_L(self):
        for i in range(100):
            N = self.ring.init_N()
            L = self.ring.calculate_ring_length(N)
            FSR = self.ring.calculate_practical_FSR(N)
            if FSR > self.required_FSR and FSR < self.required_FSR * 1.05 and np.all(L < 0.1):
                break
        if i == 99:
            raise Exception("required_FSR is too big")

        print(L, FSR)
        self.L = L
        self.FSR = FSR

    def optimize_K(self, m_L):
        self.K = self.init_K()
        action = self.generate_action()

        for t in range(self.number_of_steps):
            print("step {}".format(t + 1))
            print(self.K)
            Q = [self.calc_Q(a) for a in action]
            print(np.max(Q))
            if np.max(Q) <= 0 or np.argmax(Q) == 0:
                break
            else:
                self.K = self.K + action[np.argmax(Q)]

        self.K_list[m_L].append(self.K.tolist())
        self.Q_list[m_L].append(np.max(Q))

    def generate_action(self):
        action = [np.zeros(self.number_of_rings + 1).tolist()]
        I = np.eye(self.number_of_rings + 1)
        v = np.matrix([-0.5, -0.3, -0.1, -0.05, -0.01, 0.5, 0.3, 0.1, 0.05, 0.01])
        for i in range(self.number_of_rings + 1):
            a_i = v.T * I[i]
            action.extend(a_i.tolist())
        action.extend([[uniform(-1, 1) for _ in range(self.number_of_rings + 1)] for _ in range(5)])

        return action

    def calc_Q(self, a):
        if np.all(np.where((self.K + a > 0) & (self.K + a < self.eta), True, False)):
            mrr = self.TransferFunction(self.L, self.K + a)
            x = self.ring.calculate_x(self.FSR)
            y = mrr.simulate(x)
            evaluator = self.Evaluator(x, y)
            result = evaluator.evaluate_band()
        else:
            result = 0
        return result

    def train(self):
        for m_L in range(self.number_of_episodes_in_L):
            self.optimize_L()

            self.K_list.append([])
            self.Q_list.append([])

            for m_K in range(self.number_of_episodes_in_K):
                print("episode {}-{}".format(m_L + 1, m_K + 1))
                self.optimize_K(m_L)
            self.L_list.append(self.L)
            max_index = np.argmax(self.Q_list[m_L])
            self.Q_list[m_L] = self.Q_list[m_L][max_index]
            self.K_list[m_L] = self.K_list[m_L][max_index]

        L = self.L_list[np.argmax(self.Q_list)]
        K = self.K_list[np.argmax(self.Q_list)]
        mrr = self.TransferFunction(L, K)
        x = self.ring.calculate_x(self.FSR)
        y = mrr.simulate(x)
        evaluator = self.Evaluator(x, y)
        result = evaluator.evaluate_band()
        print(result)
        print("end")
        if result > 0:
            mrr.print_parameters()
            plot(x, y, L.size, self.logger.generate_image_path())
