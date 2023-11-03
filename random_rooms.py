import random

# Function to generate a random room
def generate_random_room(room_number):
    return {
        "name": f"Room {room_number}",
        "light": random.randint(20, 60),
        "wboard": random.choice([True, False]),
        "TV": random.choice([True, False])
    }

# Function to generate a random person
def generate_random_person(name):
    return {
        "name": name,
        "light_pref": random.randint(20, 60),
        "wboard_pref": random.choice([True, False]),
        "TV_pref": random.choice([True, False])
    }

# Generate multiple rooms
num_rooms = 10
rooms = [generate_random_room(room_number) for room_number in range(201, 201 + num_rooms)]

# Generate multiple people
num_people = 10
people_names = ["Lola", "Jason", "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Isaac", "Julia"]
people = [generate_random_person(name) for name in people_names]

# Print the generated data
print("Rooms:")
for room in rooms:
    print(room)

print("\nPeople:")
for person in people:
    print(person)
