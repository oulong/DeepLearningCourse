#conding:utf-8

import pandas as pd

'''
output = 1, if  weight1*input1 + weight2*input2 + bias >= 0
or
output = 0, if  weight1*input1 + weight2*input2 + bias < 0
'''

# wight1 = 1.0
# wight2 = 1.0
# bias = -1.1

wight1 = 1.0
wight2 = 1.0
bias = -0.1

test_inputs = [(0,0), (0,1), (1,0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []


for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = wight1 * test_input[0] + wight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_corrent_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_corrent_string])


num_wrong = len([output[4] for output in outputs if output[4]=='No'])
output_frame = pd.DataFrame(outputs, columns=['Input1', '   Input2', '  Linear_combination', '  Activation Output', ' Is Corrent'])

if not num_wrong:
    print('Nice! You are got it all corrent\n')
else:
    print('You got {} wrong. Keep trying!\n'.format(num_wrong))

print(output_frame.to_string(index=False))
