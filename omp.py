import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def omp(y, A, sparsity):
    m,N = A.shape
    res= y
    set_ys= np.zeros((m, 1))
    indices=[]
    A/=np.linalg.norm(A, ord=2, axis=0)
    solution= np.zeros((A.shape[1],1))

    for i in tqdm(range(sparsity), total=sparsity):
        corr_coeffs= abs(A.T@y)
        index= np.argmax(corr_coeffs)
        while index in indices:
            corr_coeffs= np.delete(corr_coeffs, index)
            index= np.argmax(corr_coeffs)
        indices.append(index)
        set_ys= A[:,indices]
        solution[indices] = np.linalg.pinv(set_ys).dot(y)
        res = res - A.dot(solution)
    
    return solution

if __name__=='__main__':
    
    m= 100
    N= 400
    A= np.random.randn(m,N)
    y= np.random.randn(m,1)
    sparsity= 20
    result= omp(y, A, sparsity)
    
    plt.stem(result)
    plt.show()