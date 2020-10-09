from loan_predict import model
import numpy as np

while True:

    a_name = input('Enter Your Name ')
    a_income = input('Enter Your Income ')
    c_income = input('Enter Your Coapplicant Income ')
    l_amount = input('Enter Loan Amount ')
    l_term = input('Enter term of loan ')
    c_hist = input('Enter credit history ')


    try:
        array =np.array([[a_income, c_income, l_amount, l_term, c_hist]])
        array = array.astype(int)
        # array =np.array([[1500,1000,500,300,1]])
        array.reshape(-1, 1)
        prediction = model.predict(array)
        # print(prediction)
        if prediction[0] == 1:
            print(f'Congratulations {a_name} You are eligible to take loan.')
        else:
            print(f'Sorry {a_name} You cannot take loan')
    except ValueError:
        print('Please Provide valid inputs')

    confirm = input('Do you want to check again (Write yes or no) ')
    if confirm.lower() == 'no':
        break