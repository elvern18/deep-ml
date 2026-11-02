import torch

def lu_decomposition(A) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists or torch.Tensor
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	if not isinstance(A, torch.Tensor):
		A_tensor = torch.as_tensor(A, dtype=torch.float64)
	else:
		A_tensor = A.to(torch.float64)
	
	n = A_tensor.shape[0]
	L = torch.zeros((n, n), dtype=torch.float64)
	U = torch.zeros((n, n), dtype=torch.float64)
	
	for i in range(n):
		# Upper triangular matrix U
		for k in range(i, n):
			s = sum(L[i][j] * U[j][k] for j in range(i))
			U[i][k] = A_tensor[i][k] - s
		
		# Lower triangular matrix L (1s on diagonal)
		L[i][i] = 1.0
		for k in range(i + 1, n):
			s = sum(L[k][j] * U[j][i] for j in range(i))
			L[k][i] = (A_tensor[k][i] - s) / U[i][i]
	
	return L, U
