import json
from tqdm import tqdm

with open("/home/charlie/.objathor-assets/2023_09_23/annotations.json", "r") as f:
    main_json = json.load(f)

# Initialize empty lists to store the properties
primary_properties = []
secondary_properties = []

# Iterate through each object in the main_json
for obj in tqdm(main_json.values()):
    # Extract the primaryProperty and secondaryProperties
    primary_property = obj.get("thor_metadata", {}).get("assetMetadata", {}).get("primaryProperty")
    secondary_props = obj.get("thor_metadata", {}).get("assetMetadata", {}).get("secondaryProperties", [])

    # Add the primaryProperty to the list if it exists
    if primary_property:
        primary_properties.append(primary_property)

    # Add the secondaryProperties to the list if they exist
    if secondary_props:
        secondary_properties.extend(secondary_props)

# Remove duplicates by converting the lists to sets
primary_properties = set(primary_properties)
secondary_properties = set(secondary_properties)

# Print the results
print("Primary Properties:", primary_properties)
print("Secondary Properties:", secondary_properties)