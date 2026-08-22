import xml.etree.ElementTree as ET
import re

def parse_telecom_kml(kml_path):
    """
    Parses a KML file and extracts site codes, names, latitudes, longitudes, and altitudes.
    Ensures that site names/codes are separated from numerical coordinates.
    """
    sites = []
    
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # KML XML Namespaces
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Find all Placemark elements
        placemarks = root.findall('.//kml:Placemark', namespaces)
        if not placemarks:
            # Fallback if namespace isn't explicitly declared in elements
            placemarks = root.findall('.//Placemark')

        for pm in placemarks:
            # 1. Extract Placemark Name
            name_elem = pm.find('kml:name', namespaces)
            if name_elem is None:
                name_elem = pm.find('name')
                
            raw_name = name_elem.text.strip() if name_elem is not None and name_elem.text else "UNKNOWN_SITE"

            # Filter out raw coordinates if the name itself is just a float/coordinate string
            try:
                float(raw_name.replace(',', '').strip())
                # If name is purely numeric, fallback placeholder
                site_code = f"SITE-{len(sites)+1}"
            except ValueError:
                site_code = raw_name

            # 2. Extract Coordinates (<coordinates> lon,lat,alt </coordinates>)
            coords_elem = pm.find('.//kml:coordinates', namespaces)
            if coords_elem is None:
                coords_elem = pm.find('.//coordinates')

            lat, lon, alt = None, None, 0.0

            if coords_elem is not None and coords_elem.text:
                coords_str = coords_elem.text.strip()
                # Split coordinate values (KML standard: lon, lat, alt)
                parts = re.split(r'[\s,]+', coords_str)
                parts = [p for p in parts if p]
                
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        if len(parts) >= 3:
                            alt = float(parts[2])
                    except ValueError:
                        pass

            if lat is not None and lon is not None:
                sites.append({
                    'site_code': site_code,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                })

    except Exception as e:
        print(f"Error parsing KML: {e}")

    return sites
