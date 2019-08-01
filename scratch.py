import pymongo
from bson.objectid import ObjectId


connection = pymongo.Connection()

db = connection["tutorial"]
employees = db["employees"]

employees.insert({"name": "Lucas Hightower", 'gender':'m', 'phone':'520-555-1212', 'age':8})

cursor = db.employees.find()
for employee in db.employees.find():
    print employee


print employees.find({"name":"Rick Hightower"})[0]


cursor = employees.find({"age": {"$lt": 35}})
for employee in cursor:
     print "under 35: %s" % employee


diana = employees.find_one({"_id":ObjectId("4f984cce72320612f8f432bb")})
print "Diana %s" % diana