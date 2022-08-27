import numpy as np
from scipy.stats import multinomial


class Model:

    def __init__(self, population):
        self.N = population
    
    def init_run(self):
        raise NotImplementedError
    
    def get_derivatives(self, x, t=0):
        raise NotImplementedError


class MultiDistrictModel(Model):

    def __init__(self, district_models: list, mobility_matrix=None, mobility_index=None):
        self.district_models = district_models
        self.n_districts = len(district_models)
        self.N = [model.N for model in self.district_models]
        self.mobility_matrix = mobility_matrix
        self.mobility_index = mobility_index if mobility_index else 0
    
    def init_run(self, **kwargs):
        x0 = [model.init_run(**{key:item[i] for key, item in kwargs.items()})
                for i, model in enumerate(self.district_models)]
        x0 = np.concatenate(x0, 0)

        return x0
    
    def get_derivatives(self, x, t=0):
        x = x.reshape(self.n_districts, -1)

        dx = [model.get_derivatives(x[i, :], t).reshape(1, -1) for i, model in enumerate(self.district_models)]
        dx = np.concatenate(dx, 0)

        if self.mobility_matrix is not None:
            dx = self.apply_mobility(dx, x)

        return dx.reshape(-1)
    
    def apply_mobility(self, dx, x):
        d = dx[:, self.mobility_index].reshape(-1)
        x_ = x[:, self.mobility_index].reshape(-1)

        for i in range(len(d)):
            for j in range(len(d)):
                if i == j:
                    d[i] = d[i] - x_[i] * (1 - self.mobility_matrix[i, i])
                else:
                    d[j] = d[j] + x_[i] * self.mobility_matrix[i, j]
        
        dx[:, self.mobility_index] = d

        return dx


class SEIRDV_old(Model):
    #Initialisation
    def __init__(self, r0, infect, v_rate, v_eff, asympt_infect, p_asympt, incubation, asymp_rec,
                 mild_rec, hosp_rec, icu_rec, p_asymp_mild, p_mild_hosp, p_hosp_icu, hosp_die,
                 icu_die, population, feedback_loop=None):

        self.beta_s = r0 / infect
        self.beta_v = self.beta_s * (1 - v_eff)
        self.upsilon = v_rate
        self.rho = asympt_infect
        self.p_asympt = p_asympt
        self.sigma = 1 / incubation

        self.gamma_i1 = asymp_rec
        self.gamma_i2 = mild_rec
        self.gamma_i3 = hosp_rec
        self.gamma_i4 = icu_rec

        self.alpha_i1 = p_asymp_mild
        self.alpha_i2 = p_mild_hosp
        self.alpha_i3 = p_hosp_icu

        self.delta_i3 = hosp_die
        self.delta_i4 = icu_die

        self.N = population

        self.mu = feedback_loop
    #Calculating the nunber of initial people in each compartment-- here only 1 person in exposed
    def init_run(self, sus_0=0, vac_0=0, exp_0=1, asymp_0=0, mild_0=0, hosp_0=0, icu_0=0, rec_0=0, die_0=0):
        sus_0 = self.N - vac_0 - exp_0 - asymp_0 - mild_0 - hosp_0 - icu_0 - rec_0 - die_0
        self.x0 = np.array([sus_0, vac_0, exp_0, asymp_0, mild_0, hosp_0, icu_0, rec_0, die_0])

        return self.x0
    #calculating the derivatives using the system of equations defined in paper 7
    def get_derivatives(self, x, t=0):
        s, v, e, i1, i2, i3, i4, r, d = x
        dx = np.zeros(x.shape[0])
        
        N = s + v + e + i1 + i2 + i3 + i4 + r + d
        dx[0] = -self.upsilon * s - (self.beta_s * s * (self.rho * i1 + i2 + i3 + i4) / N)
        dx[1] = self.upsilon * s - (self.beta_v * v * (self.rho * i1 + i2 + i3 + i4) / N)
        dx[2] = (self.beta_s * s * (self.rho * i1 + i2 + i3 + i4) / N) + \
            (self.beta_v * v * (self.rho * i1 + i2 + i3 + i4) / N) - self.sigma * e
        dx[3] = self.p_asympt * self.sigma * e - self.gamma_i1 * i1 - self.alpha_i1 * i1
        dx[4] = (1 - self.p_asympt) * self.sigma * e + self.alpha_i1 * i1 - \
            self.gamma_i2 * i2 - self.alpha_i2 * i2
        dx[5] = self.alpha_i2 * i2 - self.gamma_i3 * i3 - self.alpha_i3 * i3 - self.delta_i3 * i3
        dx[6] = self.alpha_i3 * i3 - self.gamma_i4 * i4 - self.delta_i4 * i4
        dx[7] = self.gamma_i1 * i1 + self.gamma_i2 * i2 + self.gamma_i3 * i3 + self.gamma_i4 * i4
        dx[8] = self.delta_i3 * i3 + self.delta_i4 * i4
    #feedback loop if implemented
        if self.mu:
            dx[0] = dx[0] + self.mu * (r + v)
            dx[1] = dx[1] - self.mu * v
            dx[7] = dx[7] - self.mu * r

        return dx


class SEIRDV(Model):
    #Initialisation
    def __init__(self, r0, infect, v_rate, v_eff, asympt_infect, p_asympt, incubation, asymp_rec,
                 mild_rec, hosp_rec, icu_rec, p_mild_hosp, p_mild_icu, hosp_die,
                 icu_die, population):

        self.beta_s = r0 / infect
        self.beta_v = self.beta_s * (1 - v_eff)
        self.upsilon = v_rate
        self.rho = asympt_infect
        self.p_asympt = p_asympt
        self.sigma = 1 / incubation

        self.gamma_i1 = asymp_rec
        self.gamma_i2 = mild_rec
        self.gamma_i3 = hosp_rec
        self.gamma_i4 = icu_rec

        self.alpha_i23 = p_mild_hosp
        self.alpha_i24 = p_mild_icu

        self.delta_i3 = hosp_die
        self.delta_i4 = icu_die

        self.N = population

    #Calculating the nunber of initial people in each compartment-- here only 1 person in exposed
    def init_run(self, sus_0=0, vac_0=0, exp_0=1, asymp_0=0, mild_0=0, hosp_0=0, icu_0=0, rec_0=0, die_0=0):
        sus_0 = self.N - vac_0 - exp_0 - asymp_0 - mild_0 - hosp_0 - icu_0 - rec_0 - die_0
        self.x0 = np.array([sus_0, vac_0, exp_0, asymp_0, mild_0, hosp_0, icu_0, rec_0, die_0])

        return self.x0
    #calculating the derivatives using the system of equations defined in paper 7
    def get_derivatives(self, x, t=0):
        s, v, e, i1, i2, i3, i4, r, d = x
        dx = np.zeros(x.shape[0])
        
        N = s + v + e + i1 + i2 + i3 + i4 + r + d
        dx[0] = -self.upsilon * s - (self.beta_s * s * (self.rho * i1 + i2 + i3 + i4) / N)
        dx[1] = self.upsilon * s - (self.beta_v * v * (self.rho * i1 + i2 + i3 + i4) / N)
        dx[2] = (self.beta_s * s * (self.rho * i1 + i2 + i3 + i4) / N) + \
            (self.beta_v * v * (self.rho * i1 + i2 + i3 + i4) / N) - self.sigma * e
        dx[3] = self.p_asympt * self.sigma * e - self.gamma_i1 * i1
        dx[4] = (1 - self.p_asympt) * self.sigma * e - self.gamma_i2 * i2 - \
                self.alpha_i23 * i2 - self.alpha_i24 * i2
        dx[5] = self.alpha_i23 * i2 - self.gamma_i3 * i3 - self.delta_i3 * i3
        dx[6] = self.alpha_i24 * i2 - self.gamma_i4 * i4 - self.delta_i4 * i4
        dx[7] = self.gamma_i1 * i1 + self.gamma_i2 * i2 + self.gamma_i3 * i3 + self.gamma_i4 * i4
        dx[8] = self.delta_i3 * i3 + self.delta_i4 * i4

        return dx


# def seidim_equations(x, t):
#     s, im, es, ei, i1s, i2s, i1i, i2i, i3, i4, d = x
#     dx = np.zeros(11)
#     dx[0] = -upsilon * s - \
#         (beta_s * s * (rho * (i1s + i1i) + (i2s + i2i) + i3 + i4) / N) + mu * im
#     dx[1] = upsilon*s - (beta_im*im*(rho*(i1s + i1i) + (i2s + i2i) + i3 + i4) / N) - \
#         mu * im + gamma_i1s * i1s + gamma_i2s * i2s + \
#         gamma_i1i * i1i + gamma_i2i*i2i + gamma_i3*i3 + gamma_i4*i4
#     dx[2] = (beta_s*s*(rho*(i1s + i1i) + (i2s + i2i) + i3 + i4) / N) - sigma*es
#     dx[3] = (beta_im * im * (rho * (i1s + i1i) +
#              (i2s + i2i) + i3 + i4) / N) - sigma * ei
#     dx[4] = p1_s * sigma * es - gamma_i1s * i1s - alpha_i1s * i1s
#     dx[5] = (1 - p1_s) * sigma * es + alpha_i1s * i1s - \
#         gamma_i2s * i2s - alpha_i2s * i2s
#     dx[6] = p1_i * sigma * ei - gamma_i1i * i1i - alpha_i1i * i1i
#     dx[7] = (1-p1_i)*sigma*ei + alpha_i1i*i1i - gamma_i2i*i2i - alpha_i2i*i2i
#     dx[8] = alpha_i2s * i2s + alpha_i2i*i2i - \
#         gamma_i3*i3 - alpha_i3*i3 - delta_i3*i3
#     dx[9] = alpha_i3*i3 - gamma_i4*i4 - delta_i4*i4
#     dx[10] = delta_i3*i3 + delta_i4*i4
#     return dx
