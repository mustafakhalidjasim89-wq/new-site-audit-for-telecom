import os
import xml.etree.ElementTree as ET

def parse_telecom_kml(kml_path):
    if not os.path.exists(kml_path):
        data_dir = os.path.dirname(kml_path)
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.kml')]
            if files:
                kml_path = files[0]
            else:
                return []
        else:
            return []

    sites = []
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # Remove namespace prefixes for uniform searching
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
                
        placemarks = root.findall('.//Placemark')
        
        for idx, pm in enumerate(placemarks):
            name_elem = pm.find('name')
            coords_elem = pm.find('.//coordinates')
            
            raw_name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"Site_{idx+1}"
            
            lat, lon = None, None
            if coords_elem is not None and coords_elem.text:
                coords = coords_elem.text.strip().split(',')
                if len(coords) >= 2:
                    try:
                        lon = float(coords[0].strip())
                        lat = float(coords[1].strip())
                    except ValueError:
                        pass
            
            sites.append({
                'site_code': raw_name,
                'name': raw_name,
                'latitude': lat,
                'longitude': lon
            })
    except Exception as e:
        print(f"Error parsing KML: {e}")
        
    return sites
