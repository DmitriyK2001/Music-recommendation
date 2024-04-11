from model import compute_estimated_matrix, compute_svd, show_recomendations
from scipy import io

def infer(K = 50, uTest = [1, 2, 3, 4, 5]):
  data_sparse = io.hb_read('state_matrix.csv')
  urm = data_sparse
  MAX_PID = urm.shape[1]
  MAX_UID = urm.shape[0]

  U, S, Vt = compute_svd(urm, K)

  uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)

  show_recomendations(uTest)

  uTest = [0]
  #Get estimated rating for test user
  print("Predictied ratings:")
  uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)
  show_recomendations(uTest)
