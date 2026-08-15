import xml.etree.ElementTree as ET

def parse_telecom_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()

    sites = []

    # Strip XML namespaces dynamically so any KML format works
    for elem in root.iter():
        if elem.tag.endswith('Placemark'):
            name = ""
            custom_code = ""
            coords = ""

            # Iterate over child elements in the Placemark
            for child in elem.iter():
                tag = child.tag.split('}')[-1]  # remove namespace prefix
                
                if tag == 'name' and not name:
                    name = child.text.strip() if child.text else ""
                
                # Check for custom tags like <mwm:customName> or fall back to name
                if tag in ['customName', 'lang'] and child.text:
                    custom_code = child.text.strip()
                
                if tag == 'coordinates':
                    coords = child.text.strip() if child.text else ""

            # Fall back site_code to name if no custom code exists
            if not custom_code:
                custom_code = name

            # Parse longitude and latitude
            lat, lon = None, None
            if coords:
                parts = coords.split()
                first_coord = parts[0] if parts else coords
                coord_split = first_coord.split(',')
                if len(coord_split) >= 2:
                    try:
                        lon = float(coord_split[0])
                        lat = float(coord_split[1])
                    except ValueError:
                        pass

            if custom_code or (lat and lon):
                sites.append({
                    'site_code': custom_code,
                    'name': name,
                    'latitude': lat,
                    'longitude': lon
                })

    return sites
