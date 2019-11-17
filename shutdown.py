from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

gcp_credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=gcp_credentials)

request = service.instances().stop(project='majestic-disk-257314', zone='us-central1-a', instance='7273726640686567037')
response = request.execute()