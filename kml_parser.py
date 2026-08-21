import xml.etree.ElementTree as ET
import re

def parse_telecom_kml(kml_path):
    """
    Parses a KML file, extracting site names while isolating coordinates.
    If a Placemark <name> is a coordinate/number, it extracts the real Site ID 
    from <description> or assigns a structured site code.
    """
    sites = []
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        placemarks = root.findall('.//kml:Placemark', namespaces)
        if not placemarks:
            placemarks = root.findall('.//Placemark')

        for idx, pm in enumerate(placemarks):
            raw_name = ""
            name_elem = pm.find('kml:name', namespaces) or pm.find('name')
            
            if name_elem is not None and name_elem.text:
                raw_name = name_elem.text.strip()

            # Check if name is purely a floating-point coordinate (e.g. "40.9796")
            is_coordinate = False
            try:
                float(raw_name.replace(',', '').strip())
                is_coordinate = True
            except ValueError:
                is_coordinate = False

            site_code = raw_name
            # If name is a coordinate number or empty, pull site ID from description
            if is_coordinate or not raw_name:
                desc_elem = pm.find('kml:description', namespaces) or pm.find('description')
                if desc_elem is not None and desc_elem.text:
                    desc_text = desc_elem.text
                    # Extract alphanumeric site code pattern (e.g., BAG-CLS5-012, CLS-01)
                    match = re.search(r'([A-Za-z0-9]+[\-_][A-Za-z0-9\-_]+)', desc_text)
                    if match:
                        site_code = match.group(1)
                    else:
                        site_code = f"SITE-{idx+1:03d}"
                else:
                    site_code = f"SITE-{idx+1:03d}"

            # Extract Coordinates (<coordinates> lon,lat,alt </coordinates>)
            coords_elem = pm.find('.//kml:coordinates', namespaces) or pm.find('.//coordinates')
            lat, lon, alt = None, None, 0.0

            if coords_elem is not None and coords_elem.text:
                coords_str = coords_elem.text.strip()
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
