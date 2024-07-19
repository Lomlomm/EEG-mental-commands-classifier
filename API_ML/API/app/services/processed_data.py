from flask import jsonify
from . import supabase_api


def getProcessData():
    supabase = supabase_api.supabase
    try:
        # first_test = supabase.table('first-test').select('*').execute().data
        # second_test = supabase.table('second-test').select('*').execute().data
        # third_test = supabase.table('third-test').select('*').execute().data
        # fourth_test = supabase.table('fourth-test').select('*').execute().data
        # fifth_test = supabase.table('fifth-test').select('*').execute().data
        concatenated_data = supabase.table('concatenated-data').select('*').execute().data


        status = 200
        message = 'Data fetch successfully'
        response = {
            'concatenated-data': concatenated_data, 
        }

    except Exception as e: 
        status = 500
        message = 'There was an issue trying to fetch the data :('
        response = str(e)
    
    return jsonify({
        'status': status, 
        'message': message, 
        'response': response
    })

def getEvaluationData():
    supabase = supabase_api.supabase
    try:
        # first_test = supabase.table('first-test').select('*').execute().data
        # second_test = supabase.table('second-test').select('*').execute().data
        # third_test = supabase.table('third-test').select('*').execute().data
        # fourth_test = supabase.table('fourth-test').select('*').execute().data
        # fifth_test = supabase.table('fifth-test').select('*').execute().data
        prueba_3 = supabase.table('prueba-3-2').select('*').execute().data


        status = 200
        message = 'Data fetch successfully'
        response = {
            'prueba-3': prueba_3
        }

    except Exception as e: 
        status = 500
        message = 'There was an issue trying to fetch the data :('
        response = str(e)
    
    return jsonify({
        'status': status, 
        'message': message, 
        'response': response
    })

if __name__ == '__main__':
    getProcessData()