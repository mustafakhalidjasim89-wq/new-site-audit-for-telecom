import xml.etree.ElementTree as ET
from geo_utils import calculate_distance

def load_kml_sites(kml_file_path: str) -> list:
    """
    Parses KML file and returns a list of dictionaries with site names and coordinates.
    """
    tree = ET.parse(kml_file_path)
    root = tree.getroot()

    # KML namespaces
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    sites = []

    for placemark in root.findall('.//kml:Placemark', ns):
        name_elem = placemark.find('kml:name', ns)
        coord_elem = placemark.find('.//kml:coordinates', ns)

        if name_elem is not None and coord_elem is not None:
            name = name_elem.text.strip()
            # KML coordinates are formatted as: longitude,latitude,altitude
            coords_str = coord_elem.text.strip().split(',')
            lon = float(coords_str[0])
            lat = float(coords_str[1])
            
            sites.append({'name': name, 'lat': lat, 'lon': lon})

    return sites


def find_nearby_sites(user_lat: float, user_lon: float, sites: list, max_radius_km: float = 5.0) -> list:
    """
    Filters and sorts sites based on user's current GPS location.
    """
    matched_sites = []
    
    for site in sites:
        dist = calculate_distance(user_lat, user_lon, site['lat'], site['lon'])
        if dist <= max_radius_km:
            matched_sites.append({
                'name': site['name'],
                'distance_km': round(dist, 2),
                'distance_m': int(dist * 1000)
            })

    # Sort nearest sites first
    matched_sites.sort(key=lambda x: x['distance_km'])
    return matched_sites
