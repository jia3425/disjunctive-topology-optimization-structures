"""OC warm start only; final acceptance always belongs to the full-space NLP.

Vectorized assembly uses the same Q4 matrices and density filter as tpo.fea.
No thresholding, new objective term, or changed physical model is introduced.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class ProjectedResponse:
    def __init__(self, data, beta, eta, penalty):
        self.data, self.beta, self.eta, self.penalty = data, beta, eta, penalty
        mesh = data.mesh
        self.n = mesh.element_count
        rows, cols, weights = [], [], []
        for e, entries in enumerate(data.filter_data.rows):
            for j, w in entries:
                rows.append(e); cols.append(j); weights.append(w)
        self.H = coo_matrix((weights, (rows, cols)), shape=(self.n, self.n)).tocsr()
        mapping = np.full(mesh.total_dofs, -1, dtype=int)
        mapping[mesh.free_dofs] = np.arange(len(mesh.free_dofs))
        dofs = mapping[mesh.element_dofs]
        size = dofs.shape[1]
        rr = np.repeat(dofs, size, axis=1).ravel()
        cc = np.tile(dofs, (1, size)).ravel()
        mask = (rr>=0)&(cc>=0)
        self.rr, self.cc = rr[mask], cc[mask]
        self.e = np.repeat(np.arange(self.n), size*size)[mask]
        self.ke = np.tile(data.element_ke.ravel(), self.n)[mask]
        self.shape = (len(mesh.free_dofs), len(mesh.free_dofs))

    def project(self, raw):
        filtered = self.H@raw
        if self.beta == 0:
            return filtered, np.ones(self.n)
        a = np.tanh(self.beta*self.eta)
        denom = a + np.tanh(self.beta*(1-self.eta))
        t = np.tanh(self.beta*(filtered-self.eta))
        return (a+t)/denom, self.beta*(1-t*t)/denom

    def evaluate(self, raw):
        physical, derivative = self.project(raw)
        cfg, mesh = self.data.config, self.data.mesh
        factors = cfg.emin+(cfg.e0-cfg.emin)*physical**self.penalty
        matrix = coo_matrix((self.ke*factors[self.e], (self.rr,self.cc)), shape=self.shape).tocsr()
        full_u = np.zeros(mesh.total_dofs)
        full_u[mesh.free_dofs] = spsolve(matrix, mesh.force[mesh.free_dofs])
        if not np.isfinite(full_u).all():
            raise RuntimeError('Nonfinite OC initializer displacement')
        local = full_u[mesh.element_dofs]
        energy = np.einsum('ei,ij,ej->e', local, self.data.element_ke, local)
        dc = self.H.T@(-self.penalty*(cfg.e0-cfg.emin)*physical**(self.penalty-1)*energy*derivative)
        dv = self.H.T@(derivative/self.n)
        return float(mesh.force@full_u), float(physical.mean()), dc, dv, physical


def warm_start(data, raw, beta, eta, penalty, max_iter=400, move=.05):
    response = ProjectedResponse(data, beta, eta, penalty)
    cfg = data.config
    raw = raw.copy()
    trace = []
    for iteration in range(1, max_iter+1):
        c, volume, dc, dv, physical = response.evaluate(raw)
        lo = np.maximum(cfg.xmin, raw-move)
        hi = np.minimum(1., raw+move)

        def candidate(multiplier):
            ratio = np.maximum(1e-30, -dc/np.maximum(dv*multiplier, 1e-30))
            return np.clip(raw*np.sqrt(ratio), lo, hi)

        lower, upper = 0., 1.
        # A newly changed projection can make even the lowest move infeasible.
        # In that case take the maximal permitted volume-reducing step.
        if response.project(lo)[0].mean() > cfg.volfrac:
            updated = lo
        else:
            while response.project(candidate(upper))[0].mean() > cfg.volfrac:
                upper *= 2
                if upper > 1e30:
                    raise RuntimeError('Cannot bracket OC multiplier')
            updated = candidate(upper)
            for _ in range(60):
                middle = (lower+upper)/2
                trial = candidate(middle)
                if response.project(trial)[0].mean() > cfg.volfrac:
                    lower = middle
                else:
                    upper, updated = middle, trial
                if (upper-lower)/max(upper, 1e-30) < 1e-10:
                    break
        change = float(np.max(np.abs(updated-raw)))
        trace.append({'iteration':iteration,'compliance_before_update':c,
                      'volume_before_update':volume,'grayness_before_update':float(np.mean(4*physical*(1-physical))),
                      'max_raw_change':change})
        raw = updated
        if iteration%50 == 0:
            print(f'OC_INITIALIZER iter={iteration} C={c:.8g} gray={trace[-1]["grayness_before_update"]:.6g} change={change:.3g}', flush=True)
        if change < 1e-5:
            break
    return raw, trace
