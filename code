# sample rooms with attributes
rooms = [
    {"name": "Room 202", "light": 30, "wboard": True, "TV": False},
    {"name": "Room 350", "light": 40, "wboard": False, "TV": True},
    {"name": "Room 211", "light": 35, "wboard": True, "TV": False},
    
]

# sample people with their preferences
# compute distance between light preferences 
# light preference is an interval between 30-40, if value is inside interval give a one. 
# several rooms are equally a match, 
people = [
    {"name": "Lola", "light_pref": 35, "wboard_pref": True, "TV_pref": False},
    {"name": "Jason", "light_pref": 40, "wboard_pref": False, "TV_pref": True},
    
]

def calculate_room_score(room, person):
    # Calculate a score based on preferences and room attributes
    score = 0

    if room["light"] == person["light_pref"]:
        score += 1

    if room["wboard"] == person["wboard_pref"]:
        score += 1

    if room["TV"] == person["TV_pref"]:
        score += 1

    return score


def match_rooms_to_people(rooms, people):
    matches = {}

    for person in people:
        best_room = None
        best_score = -1

        for room in rooms:
            score = calculate_room_score(room, person)

            if score > best_score:
                best_score = score
                best_room = room

        if best_room is not None:
            matches[person["name"]] = best_room["name"]

    return matches

# Find the best room matches for people
room_assignments = match_rooms_to_people(rooms, people)

# Print the room assignments
for person, room in room_assignments.items():
    print(f"{person} is assigned to {room}")

