import xml.etree.ElementTree as ET

def parse_telecom_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()

    namespaces = {
        'kml': 'http://www.opengis.net/kml/2.2',
        'mwm': 'http://mapswithme.com/kml/ext/1.0'
    }

    sites = []
    for placemark in root.findall('.//kml:Placemark', namespaces):
        name_elem = placemark.find('kml:name', namespaces)
        name = name_elem.text.strip() if name_elem is not None else ""

        custom_name_elem = placemark.find('.//mwm:customName/mwm:lang', namespaces)
        custom_code = custom_name_elem.text.strip() if custom_name_elem is not None else name

        coords_elem = placemark.find('.//kml:coordinates', namespaces)
        coords = coords_elem.text.strip() if coords_elem is not None else ""
        lon, lat, *_ = coords.split(',') if coords else (None, None)

        if lat and lon:
            sites.append({
                'site_code': custom_code,
                'name': name,
                'latitude': float(lat),
                'longitude': float(lon)
            })
    return sites
