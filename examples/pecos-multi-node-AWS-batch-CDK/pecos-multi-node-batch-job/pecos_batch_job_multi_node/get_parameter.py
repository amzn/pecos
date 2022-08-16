from enum import unique
import json
def get_parameters():
    memory_input = input("Please input the memory(numbers only) you want.\nThe default for testing is 10GB\n")
    default_memory = 10
    memory = check_user_input_for_num(memory_input, default_memory)
    print("User choosed memory is", memory)

    disk_input = input("Please input the size of disk(numbers only) you want. \nThe default for testing is 1000GB\n")
    default_disk = 1000
    disk = check_user_input_for_num(disk_input, default_disk)
    print("User choosed disk is", disk)

    vCPU_input = input("Please input number of vCPUs(numbers only) you want.\nThe default for testing is 2\n")
    default_vCPU = 2
    vCPU = check_user_input_for_num(vCPU_input, default_vCPU)
    print("User choosed vCPUs are", vCPU)


    num_nodes_input = input("Please choose number of nodes(numbers only) you want.\nThe default for testing is 2\n")
    default_nodes = 2
    num_nodes = check_user_input_for_num(num_nodes_input, default_nodes)
    print("User choosed number of nodes are", num_nodes)

    identifier_input = input("Please input your identifier for differentiating your AWS resources. This step is mandatory\n")
    identifier = check_user_input_for_str(identifier_input)
    
    print("The default key and value pair for tags are 'Createdby'+'identifier'")
    key_input = input("Please input your key for tags. The default is 'Createdby'\n")
    key = "Createdby" if not key_input else key_input

    value_input = input("Please input your value for tags. The default is your identifier.\n")
    value = identifier if not value_input else value_input

    data = {}
    data['memory'] = memory
    data['disk'] = disk
    data['vCPU'] = vCPU
    data['identifier'] = identifier
    data['key'] = key
    data['value'] = value
    data['nodes'] = num_nodes


    write_to_JSON_File('pecos_batch_job_multi_node','user_input',data)
 
def check_user_input_for_num(user_input, default):
    if not user_input:
        return default
    while True:
        val = int(user_input)
        if val:
            break
        else:
            print("Input should be an integer")
            user_input=input("please input again.\n")
    return user_input

    
def check_user_input_for_str(user_input):
    while True:
        if user_input:
                break
        else:
            print("Identifier is important to differentiate resources")
            user_input=input("please input your identifier.\n")
    return user_input


def write_to_JSON_File(path, fileName, data):
    filePathNameWExt = './'+ path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)



get_parameters()





