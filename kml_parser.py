import os
import xml.etree.ElementTree as ET

def parse_telecom_kml(kml_path):
    # إذا لم يجد الملف بالمسار المحدد، يبحث عن أي ملف ينتهي بـ .kml داخل مجلد data
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
        
        # التعامل مع الكائنات في KML
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # البحث عن كافة عناصر Placemark
        placemarks = root.findall('.//kml:Placemark', namespaces) or root.findall('.//Placemark')
        
        for pm in placemarks:
            name_elem = pm.find('kml:name', namespaces) or pm.find('name')
            coords_elem = pm.find('.//kml:coordinates', namespaces) or pm.find('.//coordinates')
            
            site_name = name_elem.text.strip() if name_elem is not None and name_elem.text else None
            
            lat, lon = None, None
            if coords_elem is not None and coords_elem.text:
                coords = coords_elem.text.strip().split(',')
                if len(coords) >= 2:
                    lon = float(coords[0])
                    lat = float(coords[1])
            
            if site_name:
                sites.append({
                    'site_code': site_name,
                    'name': site_name,
                    'latitude': lat,
                    'longitude': lon
                })
    except Exception as e:
        print(f"Error parsing KML: {e}")
        
    return sites
