import firebase_admin
from firebase_admin import credentials, db
import json

cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred, {
'databaseURL': 'https://original-dryad-251711.firebaseio.com/'
})

def listener(event):
    data = event.data
    if isinstance(data, dict):
        for key in data:
            print(data[key])
        # with open('train-stats.json', 'a') as file:
        #     json.dump(data, file, indent=4)    
    else:
        print(data)

root = db.reference('/').listen(listener)