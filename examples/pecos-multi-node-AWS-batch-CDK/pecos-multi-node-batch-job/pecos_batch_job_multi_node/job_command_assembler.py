from enum import unique
import json
def job_command_assembler():
  
    prepare_data_input = input("Please enter the command to prepare your data\n")
    print("Your input is "+ prepare_data_input)
    train_data_command = input("Please enter the command to train your data.\n")
    print("Your input is "+ train_data_command)
    predict_command = input("Please enter the command to predict.\n")
    print("Your input is "+ predict_command)
    output_path = input("Please enter the S3 bucket location to save outputs.\n")
    print("Your input is "+ output_path)

    data = {}
    data['prepare_data_input'] = prepare_data_input
    data['train_data_command'] = train_data_command
    data['predict_command'] = predict_command
    data['output_path'] = output_path

    write_to_JSON_File('pecos_batch_job_multi_node','job_command',data)
 
def write_to_JSON_File(path, fileName, data):
    filePathNameWExt = './'+ path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)

job_command_assembler()