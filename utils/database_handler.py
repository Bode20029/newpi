from pymongo import MongoClient
from gridfs import GridFS
import cv2
from config import EV_BRANDS
from bson.objectid import ObjectId

class DatabaseHandler:
    def __init__(self):
        # Hardcoded connection details
        self.mongo_uri = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
        self.db_name = "cardb"
        self.collection_name = "nonev"
        self.client = None
        self.db = None
        self.collection = None
        self.fs = None

    def initialize(self):
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.fs = GridFS(self.db)
            # Test the connection
            self.client.server_info()
            print("Successfully connected to MongoDB")
            return True
        except Exception as e:
            print(f"Failed to initialize MongoDB. Error: {e}")
            print(f"Attempted to connect with URI: {self.mongo_uri}")
            return False

    def save_image(self, frame, timestamp):
        _, img_encoded = cv2.imencode('.jpg', frame)
        file_id = self.fs.put(img_encoded.tobytes(), filename=f"event_{timestamp}.jpg", metadata={"timestamp": timestamp})
        return file_id

    def save_event(self, file_id, timestamp, event, vehicle_class):
        event_data = {
            "file_id": file_id,
            "timestamp": timestamp,
            "event": event,
            "vehicle_class": vehicle_class,
            "is_ev": vehicle_class in EV_BRANDS
        }
        return self.collection.insert_one(event_data)

    def get_image(self, file_id):
        return self.fs.get(ObjectId(file_id))

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")
# from pymongo import MongoClient
# from gridfs import GridFS
# import cv2
# from config import EV_BRANDS
# from bson.objectid import ObjectId
# import logging

# logger = logging.getLogger(__name__)

# class DatabaseHandler:
#     def __init__(self, mongo_uri, db_name, collection_name):
#         self.mongo_uri = mongo_uri
#         self.db_name = db_name
#         self.collection_name = collection_name
#         self.client = None
#         self.db = None
#         self.collection = None
#         self.fs = None

#     def initialize(self):
#         try:
#             self.client = MongoClient(self.mongo_uri)
#             self.db = self.client[self.db_name]
#             self.collection = self.db[self.collection_name]
#             self.fs = GridFS(self.db)
#             self.client.server_info()  # This will raise an exception if the connection fails
#             logger.info("Successfully connected to MongoDB")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to initialize MongoDB: {e}")
#             return False

#     def save_image(self, frame, timestamp):
#         try:
#             _, img_encoded = cv2.imencode('.jpg', frame)
#             file_id = self.fs.put(img_encoded.tobytes(), filename=f"event_{timestamp}.jpg", metadata={"timestamp": timestamp})
#             logger.info(f"Image saved with file_id: {file_id}")
#             return file_id
#         except Exception as e:
#             logger.error(f"Failed to save image: {e}")
#             return None

#     def save_event(self, file_id, timestamp, event, vehicle_class):
#         try:
#             event_data = {
#                 "file_id": file_id,
#                 "timestamp": timestamp,
#                 "event": event,
#                 "vehicle_class": vehicle_class,
#                 "is_ev": vehicle_class in EV_BRANDS
#             }
#             result = self.collection.insert_one(event_data)
#             logger.info(f"Event saved with id: {result.inserted_id}")
#             return result.inserted_id
#         except Exception as e:
#             logger.error(f"Failed to save event: {e}")
#             return None

#     def get_image(self, file_id):
#         try:
#             return self.fs.get(ObjectId(file_id))
#         except Exception as e:
#             logger.error(f"Failed to retrieve image with file_id {file_id}: {e}")
#             return None

#     def get_event(self, event_id):
#         try:
#             return self.collection.find_one({"_id": ObjectId(event_id)})
#         except Exception as e:
#             logger.error(f"Failed to retrieve event with id {event_id}: {e}")
#             return None

#     def close(self):
#         if self.client:
#             self.client.close()
#             logger.info("MongoDB connection closed")