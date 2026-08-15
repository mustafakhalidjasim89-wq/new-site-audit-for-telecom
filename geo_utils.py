import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth 
    in kilometers using the Haversine formula (pure Python standard library).
    """
    R = 6371.0  # Earth's radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def find_nearby_sites(lat, lon, all_sites, radius_km=5.0):
    """
    Filters a list of site dictionaries and returns those within radius_km distance.
    """
    nearby = []
    for site in all_sites:
        s_lat = site.get('latitude')
        s_lon = site.get('longitude')

        if s_lat is not None and s_lon is not None:
            dist = haversine_distance(lat, lon, s_lat, s_lon)
            if dist <= radius_km:
                site_copy = site.copy()
                site_copy['distance_km'] = round(dist, 2)
                nearby.append(site_copy)

    # Sort results from closest to farthest
    nearby.sort(key=lambda x: x['distance_km'])
    return nearby
