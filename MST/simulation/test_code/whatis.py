import numpy as np

if __name__ =="__main__":
    data = np.load('/home/lpaillet/Documents/MST/simulation/test_code/test.npy')
    print(data.shape)
    print(data[15,...])