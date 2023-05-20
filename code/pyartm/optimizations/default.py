import numpy as np
import pickle
from tqdm.notebook import tqdm
from scipy import sparse
from . import base


class Optimizer(base.Optimizer):
    __slots__ = tuple()

    def __init__(
        self,
        regularization_list=None,
        loss_function=None,
        return_counters=False,
        const_phi=False,
        const_theta=False,
        inplace=False,
        verbose=False,
        iteration_callback=None
    ):
        super(Optimizer, self).__init__(
            regularization_list=regularization_list,
            loss_function=loss_function,
            return_counters=return_counters,
            const_phi=const_phi,
            const_theta=const_theta,
            inplace=inplace,
            verbose=verbose,
            iteration_callback=iteration_callback
        )

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix, tau ):
        n_tw, n_dt = None, None 
        for it in tqdm(range(self.iters_count)):
            phi_matrix_tr = np.transpose(phi_matrix)

            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr
            )

            if (tau is not None):
              EPS = 1/(2**(15))
              p_t = np.sum(theta_matrix, axis = 0)/(np.sum(theta_matrix) + EPS)
              p_t_div = 1/(p_t + EPS)
              b_dw = np.zeros((10000,2000))
              for d in range(2000):
                new_phi = phi_matrix
                probs = np.multiply((new_phi.transpose()),theta_matrix[d])
                probs = np.divide(probs.transpose(), np.sum(new_phi ,axis = 0) + EPS)
                probs = np.multiply(probs.transpose(), p_t_div)
                b_dw[:,d] = np.array(np.sum(probs, axis = 1))
              A = sparse.csr_matrix.multiply(A, sparse.csr_matrix(b_dw.transpose()*(tau) + 1))

            n_dt = A.dot(phi_matrix_tr) * theta_matrix
            n_tw = np.transpose(
                A.tocsc().transpose().dot(theta_matrix)) * phi_matrix

            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )
            n_tw += r_tw
            n_dt += r_dt

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )
            #if (it % 100 == 0):
              #with open('drive/MyDrive/phi5000w_5000d_50t_div2_{}.pkl'.format(it), 'wb') as resource_file:
                  #pickle.dump(phi_matrix, resource_file)
              #with open('drive/MyDrive/theta5000w_5000d_50t_div2_{}.pkl'.format(it), 'wb') as resource_file:
                  #pickle.dump(theta_matrix, resource_file)
              
        #with open('phi_test', 'wb') as resource_file:
            #pickle.dump(phi_matrix, resource_file)
        return phi_matrix, theta_matrix, n_tw, n_dt
