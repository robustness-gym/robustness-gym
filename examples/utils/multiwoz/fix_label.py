
def fix_general_label_error(labels, type, slots, ontology_version=""):
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels]) 

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house","guesthouses":"guest house","guest":"guest house","mutiple sports":"multiple sports", 
        "mutliple sports":"multiple sports","sports":"multiple sports","swimmingpool":"swimming pool", 
        "concerthall":"concert hall", "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", 
        "colleges":"college", "coll":"college","architectural":"architecture", "musuem":"museum", "churches":"church",
        
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", 
        "cen":"centre", "east side":"east","east area":"east", "west part of town":"west", "ce":"centre",  
        "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", 
        "north part of town":"north", "centre of town":"centre", "cb30aq": "none",
        
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        
        # day
        "monda": "monday", 
        
        # parking
        "free parking":"free",
        
        # internet
        "free internet":"yes",
        
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        
        # others 
        "y":"yes", "any":"do n't care", "n":"no", "does not care":"do n't care", "not men":"none", "not":"none", 
        "not mentioned":"none", '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
    }

    for slot in slots:
        if slot in label_dict.keys():
            
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
            
            # do not care
            if label_dict[slot] in ["doesn't care", "don't care", "dont care", "does not care", "do not care", "dontcare"]:
                label_dict[slot] = "do n't care"
            
            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"]:
                label_dict[slot] = "none"
            if slot == "hotel-internet" and label_dict[slot] == "4":
                label_dict[slot] = "none"
            if slot == "hotel-internet" and label_dict[slot] == "4":
                label_dict[slot] = "none"
            if slot == "hotel-pricerange" and label_dict[slot] == "2":
                label_dict[slot] = "none"
            if "area" in slot and label_dict[slot] in ["moderate"]:
                label_dict[slot] = "none"
            if "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            if slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            if slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            # if slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"]:
            #   label_dict[slot] = "none"
            
            if "area" in slot:
                if label_dict[slot] == "no": 
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we": 
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent": 
                    label_dict[slot] = "centre"
            
            if "day" in slot:
                if label_dict[slot] == "we": 
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            
            if "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            if "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            if ontology_version in ["1.0"]:
                # Add on May, 2020
                label_dict[slot] = label_dict[slot].replace("theater", "theatre").replace("guesthouse", "guest house")

                # Typo or naming
                if label_dict[slot] == "cafe uno": 
                    label_dict[slot] = "caffe uno"
                if label_dict[slot] == "alpha milton guest house": 
                    label_dict[slot] = "alpha-milton guest house"
                if label_dict[slot] in ["churchills college", "churchhill college", "churchill", "the churchill college"]: 
                    label_dict[slot] = "churchill college"
                if label_dict[slot] == "portugese": 
                    label_dict[slot] = "portuguese"
                if label_dict[slot] == "pizza hut fenditton": 
                    label_dict[slot] = "pizza hut fen ditton"
                if label_dict[slot] == "restaurant 17": 
                    label_dict[slot] = "restaurant one seven"
                if label_dict[slot] == "restaurant 2 two": 
                    label_dict[slot] = "restaurant two two"
                if label_dict[slot] == "gallery at 12 a high street": 
                    label_dict[slot] = "gallery at twelve a high street"
                if label_dict[slot] == "museum of archaelogy": 
                    label_dict[slot] = "museum of archaelogy and anthropology"
                if label_dict[slot] in ["huntingdon marriot hotel", "marriot hotel"]: 
                    label_dict[slot] = "huntingdon marriott hotel"
                if label_dict[slot] in ["sheeps green and lammas land park fen causeway", "sheeps green and lammas land park"]: 
                    label_dict[slot] = "sheep's green and lammas land park fen causeway"
                if label_dict[slot] in ["cambridge and country folk museum", "county folk museum"]: 
                    label_dict[slot] = "cambridge and county folk museum"
                if label_dict[slot] == "ambridge": 
                    label_dict[slot] = "cambridge"
                if label_dict[slot] == "cambridge contemporary art museum": 
                    label_dict[slot] = "cambridge contemporary art"    
                if label_dict[slot] == "molecular gastonomy": 
                    label_dict[slot] = "molecular gastronomy"
                if label_dict[slot] == "2 two and cote": 
                    label_dict[slot] = "two two and cote"
                if label_dict[slot] == "caribbeanindian": 
                    label_dict[slot] = "caribbean|indian"
                if label_dict[slot] == "whipple museum": 
                    label_dict[slot] = "whipple museum of the history of science"
                if label_dict[slot] == "ian hong": 
                    label_dict[slot] = "ian hong house"
                if label_dict[slot] == "sundaymonday": 
                    label_dict[slot] = "sunday|monday"
                if label_dict[slot] == "mondaythursday": 
                    label_dict[slot] = "monday|thursday"
                if label_dict[slot] == "fridaytuesday": 
                    label_dict[slot] = "friday|tuesday"
                if label_dict[slot] == "cheapmoderate": 
                    label_dict[slot] = "cheap|moderate"
                if label_dict[slot] == "golden house                            golden house": 
                    label_dict[slot] = "the golden house"  
                if label_dict[slot] == "golden house": 
                    label_dict[slot] = "the golden house" 
                if label_dict[slot] == "sleeperz": 
                    label_dict[slot] = "sleeperz hotel"
                if label_dict[slot] == "jamaicanchinese": 
                    label_dict[slot] = "jamaican|chinese"
                if label_dict[slot] == "shiraz": 
                    label_dict[slot] = "shiraz restaurant"
                if label_dict[slot] == "museum of archaelogy and anthropogy": 
                    label_dict[slot] = "museum of archaelogy and anthropology"    
                if label_dict[slot] == "yipee noodle bar": 
                    label_dict[slot] = "yippee noodle bar"
                if label_dict[slot] == "abc theatre": 
                    label_dict[slot] = "adc theatre"
                if label_dict[slot] == "wankworth house": 
                    label_dict[slot] = "warkworth house"
                if label_dict[slot] in ["cherry hinton water play park", "cherry hinton water park"]: 
                    label_dict[slot] = "cherry hinton water play"
                if label_dict[slot] == "the gallery at 12": 
                    label_dict[slot] = "the gallery at twelve"
                if label_dict[slot] == "barbequemodern european": 
                    label_dict[slot] = "barbeque|modern european"
                if label_dict[slot] == "north americanindian": 
                    label_dict[slot] = "north american|indian"
                if label_dict[slot] == "chiquito": 
                    label_dict[slot] = "chiquito restaurant bar"
                    

                # Abbreviation
                if label_dict[slot] == "city centre north bed and breakfast": 
                    label_dict[slot] = "city centre north b and b"
                if label_dict[slot] == "north bed and breakfast": 
                    label_dict[slot] = "north b and b"

                # Article and 's
                if label_dict[slot] == "christ college": 
                    label_dict[slot] = "christ's college"
                if label_dict[slot] == "kings college": 
                    label_dict[slot] = "king's college"
                if label_dict[slot] == "saint johns college": 
                    label_dict[slot] = "saint john's college"
                if label_dict[slot] == "kettles yard": 
                    label_dict[slot] = "kettle's yard"
                if label_dict[slot] == "rosas bed and breakfast": 
                    label_dict[slot] = "rosa's bed and breakfast"
                if label_dict[slot] == "saint catharines college": 
                    label_dict[slot] = "saint catharine's college"  
                if label_dict[slot] == "little saint marys church": 
                    label_dict[slot] = "little saint mary's church"
                if label_dict[slot] == "great saint marys church": 
                    label_dict[slot] = "great saint mary's church"
                if label_dict[slot] in ["queens college", "queens' college"]: 
                    label_dict[slot] = "queen's college"
                if label_dict[slot] == "peoples portraits exhibition at girton college": 
                    label_dict[slot] = "people's portraits exhibition at girton college"
                if label_dict[slot] == "st johns college": 
                    label_dict[slot] = "saint john's college"
                if label_dict[slot] == "whale of time": 
                    label_dict[slot] = "whale of a time"
                if label_dict[slot] in ["st catharines college", "saint catharines college"]: 
                    label_dict[slot] = "saint catharine's college"   

                # Time
                if label_dict[slot] == "16,15": 
                    label_dict[slot] = "16:15"
                if label_dict[slot] == "1330": 
                    label_dict[slot] = "13:30"
                if label_dict[slot] == "1430": 
                    label_dict[slot] = "14:30"
                if label_dict[slot] == "1532": 
                    label_dict[slot] = "15:32"
                if label_dict[slot] == "845": 
                    label_dict[slot] = "08:45"
                if label_dict[slot] == "1145": 
                    label_dict[slot] = "11:45"
                if label_dict[slot] == "1545": 
                    label_dict[slot] = "15:45"
                if label_dict[slot] == "1329": 
                    label_dict[slot] = "13:29"
                if label_dict[slot] == "1345": 
                    label_dict[slot] = "13:45"
                if label_dict[slot] == "1715": 
                    label_dict[slot] = "17:15"
                if label_dict[slot] == "929": 
                    label_dict[slot] = "09:29"


                # restaurant
                if slot == "restaurant-name" and "meze bar" in label_dict[slot]: 
                    label_dict[slot] = "meze bar restaurant"
                if slot == "restaurant-name" and label_dict[slot] == "alimentum": 
                    label_dict[slot] = "restaurant alimentum"
                if slot == "restaurant-name" and label_dict[slot] == "good luck": 
                    label_dict[slot] = "the good luck chinese food takeaway"
                if slot == "restaurant-name" and label_dict[slot] == "grafton hotel": 
                    label_dict[slot] = "grafton hotel restaurant"
                if slot == "restaurant-name" and label_dict[slot] == "2 two": 
                    label_dict[slot] = "restaurant two two"   
                if slot == "restaurant-name" and label_dict[slot] == "hotpot": 
                    label_dict[slot] = "the hotpot"   
                if slot == "restaurant-name" and label_dict[slot] == "hobsons house": 
                    label_dict[slot] = "hobson house"       
                if slot == "restaurant-name" and label_dict[slot] == "shanghai": 
                    label_dict[slot] = "shanghai family restaurant"
                if slot == "restaurant-name" and label_dict[slot] == "17": 
                    label_dict[slot] = "restaurant one seven"
                if slot == "restaurant-name" and label_dict[slot] in ["22", "restaurant 22"]: 
                    label_dict[slot] = "restaurant two two"
                if slot == "restaurant-name" and label_dict[slot] == "the maharajah tandoor": 
                    label_dict[slot] = "maharajah tandoori restaurant"
                if slot == "restaurant-name" and label_dict[slot] == "the grafton hotel": 
                    label_dict[slot] = "grafton hotel restaurant"
                if slot == "restaurant-name" and label_dict[slot] == "gardenia": 
                    label_dict[slot] = "the gardenia"
                if slot == "restaurant-name" and label_dict[slot] == "el shaddia guest house": 
                    label_dict[slot] = "el shaddai"   
                if slot == "restaurant-name" and label_dict[slot] == "the bedouin": 
                    label_dict[slot] = "bedouin"
                if slot == "restaurant-name" and label_dict[slot] == "the kohinoor": 
                    label_dict[slot] = "kohinoor"
                if slot == "restaurant-name" and label_dict[slot] == "the peking": 
                    label_dict[slot] = "peking restaurant" 
                if slot == "restaurant-book time" and label_dict[slot] == "7pm": 
                    label_dict[slot] = "19:00"
                if slot == "restaurant-book time" and label_dict[slot] == "4pm": 
                    label_dict[slot] = "16:00"
                if slot == "restaurant-book time" and label_dict[slot] == "8pm": 
                    label_dict[slot] = "20:00"
                if slot == "restaurant-name" and label_dict[slot] == "sitar": 
                    label_dict[slot] = "sitar tandoori"
                if slot == "restaurant-name" and label_dict[slot] == "binh": 
                    label_dict[slot] = "thanh binh"
                if slot == "restaurant-name" and label_dict[slot] == "mahal": 
                    label_dict[slot] = "mahal of cambridge" 

                # attraction
                if slot == "attraction-name" and label_dict[slot] == "scudamore": 
                    label_dict[slot] = "scudamores punting co"
                if slot == "attraction-name" and label_dict[slot] == "salsa": 
                    label_dict[slot] = "club salsa"
                if slot == "attraction-name" and label_dict[slot] in ["abbey pool", "abbey pool and astroturf"]: 
                    label_dict[slot] = "abbey pool and astroturf pitch"
                if slot == "attraction-name" and label_dict[slot] == "cherry hinton hall": 
                    label_dict[slot] = "cherry hinton hall and grounds"
                if slot == "attraction-name" and label_dict[slot] == "trinity street college": 
                    label_dict[slot] = "trinity college"
                if slot == "attraction-name" and label_dict[slot] == "the wandlebury": 
                    label_dict[slot] = "wandlebury country park" 
                if slot == "attraction-name" and label_dict[slot] == "king hedges learner pool": 
                    label_dict[slot] = "kings hedges learner pool" 
                if slot == "attraction-name" and label_dict[slot] in ["botanic gardens", "cambridge botanic gardens"]: 
                    label_dict[slot] = "cambridge university botanic gardens"
                if slot == "attraction-name" and label_dict[slot] == "soultree": 
                    label_dict[slot] = "soul tree nightclub"
                if slot == "attraction-name" and label_dict[slot] == "queens": 
                    label_dict[slot] = "queen's college"
                if slot == "attraction-name" and label_dict[slot] == "sheeps green": 
                    label_dict[slot] = "sheep's green and lammas land park fen causeway"
                if slot == "attraction-name" and label_dict[slot] == "jesus green": 
                    label_dict[slot] = "jesus green outdoor pool" 
                if slot == "attraction-name" and label_dict[slot] == "adc": 
                    label_dict[slot] = "adc theatre"
                if slot == "attraction-name" and label_dict[slot] == "hobsons house": 
                    label_dict[slot] = "hobson house" 
                if slot == "attraction-name" and label_dict[slot] == "cafe jello museum": 
                    label_dict[slot] = "cafe jello gallery"    
                if slot == "attraction-name" and label_dict[slot] == "whippple museum": 
                    label_dict[slot] = "whipple museum of the history of science"
                if slot == "attraction-type" and label_dict[slot] == "boating": 
                    label_dict[slot] = "boat"  
                if slot == "attraction-name" and label_dict[slot] == "peoples portraits exhibition": 
                    label_dict[slot] = "people's portraits exhibition at girton college" 
                if slot == "attraction-name" and label_dict[slot] == "lammas land park": 
                    label_dict[slot] = "sheep's green and lammas land park fen causeway"

                # taxi
                if slot in ["taxi-destination", "taxi-departure"] and label_dict[slot] == "meze bar": 
                    label_dict[slot] = "meze bar restaurant"
                if slot in ["taxi-destination", "taxi-departure"] and label_dict[slot] == "el shaddia guest house": 
                    label_dict[slot] = "el shaddai"
                if slot == "taxi-departure" and label_dict[slot] == "centre of town at my hotel": 
                    label_dict[slot] = "hotel" 

                # train
                if slot == "train-departure" and label_dict[slot] in ["liverpool", "london liverpool"]: 
                    label_dict[slot] = "london liverpool street" 
                if slot == "train-destination" and label_dict[slot] == "liverpool street": 
                    label_dict[slot] = "london liverpool street"
                if slot == "train-departure" and label_dict[slot] == "alpha milton": 
                    label_dict[slot] = "alpha-milton" 

                # hotel
                if slot == "hotel-name" and label_dict[slot] == "el shaddia guest house": 
                    label_dict[slot] = "el shaddai"
                if slot == "hotel-name" and label_dict[slot] == "alesbray lodge guest house": 
                    label_dict[slot] = "aylesbray lodge guest house"   
                if slot == "hotel-name" and label_dict[slot] == "the gonvile hotel": 
                    label_dict[slot] = "the gonville hotel"    
                if slot == "hotel-name" and label_dict[slot] == "no": 
                    label_dict[slot] = "none" 
                if slot == "hotel-name" and label_dict[slot] in ["holiday inn", "holiday inn cambridge"]: 
                    label_dict[slot] = "express by holiday inn cambridge" 
                if slot == "hotel-name" and label_dict[slot] == "wartworth": 
                    label_dict[slot] = "warkworth house"   

                # Suppose to be a wrong annotation
                if slot == "restaurant-name" and label_dict[slot] == "south": 
                    label_dict[slot] = "none"
                if slot == "attraction-type" and label_dict[slot] == "churchill college": 
                    label_dict[slot] = "none"
                if slot == "attraction-name" and label_dict[slot] == "boat": 
                    label_dict[slot] = "none" 
                if slot == "attraction-type" and label_dict[slot] == "museum kettles yard": 
                    label_dict[slot] = "none"
                if slot == "attraction-type" and label_dict[slot] == "hotel": 
                    label_dict[slot] = "none"
                if slot == "attraction-type" and label_dict[slot] == "camboats": 
                    label_dict[slot] = "boat" 


                # TODO: Need to check with dialogue data to deal with strange labels before

                # if slot == "restaurant-name" and label_dict[slot] == "eraina and michaelhouse cafe": 
                #    label_dict[slot] = "eraina|michaelhouse cafe"
                # if slot == "attraction-name" and label_dict[slot] == "gonville hotel": 
                #    label_dict[slot] = "none"
                # if label_dict[slot] == "good luck": 
                #    label_dict[slot] = "the good luck chinese food takeaway"
                # if slot == "restaurant-book time" and label_dict[slot] == "9": 
                #    label_dict[slot] = "21:00"
                # if slot == "taxi-departure" and label_dict[slot] == "girton college": 
                #    label_dict[slot] = "people's portraits exhibition at girton college"
                # if slot == "restaurant-name" and label_dict[slot] == "molecular gastronomy": 
                #    label_dict[slot] = "none"
                # [Info] Adding Slot: restaurant-name with value: primavera
                # [Info] Adding Slot: train-departure with value: huntingdon
                # [Info] Adding Slot: attraction-name with value: aylesbray lodge guest house
                # [Info] Adding Slot: attraction-name with value: gallery
                # [Info] Adding Slot: hotel-name with value: eraina
                # [Info] Adding Slot: restaurant-name with value: india west
                # [Info] Adding Slot: restaurant-name with value: autumn house
                # [Info] Adding Slot: train-destination with value: norway
                # [Info] Adding Slot: attraction-name with value: cinema cinema
                # [Info] Adding Slot: hotel-name with value: lan hon
                # [Info] Adding Slot: restaurant-food with value: sushi
                # [Info] Adding Slot: attraction-name with value: university arms hotel
                # [Info] Adding Slot: train-departure with value: stratford
                # [Info] Adding Slot: attraction-name with value: history of science museum
                # [Info] Adding Slot: restaurant-name with value: nil
                # [Info] Adding Slot: train-leaveat with value: 9
                # [Info] Adding Slot: restaurant-name with value: ashley hotel
                # [Info] Adding Slot: taxi-destination with value: the cambridge shop
                # [Info] Adding Slot: hotel-name with value: acorn place
                # [Info] Adding Slot: restaurant-name with value: de luca cucina and bar riverside brasserie
                # [Info] Adding Slot: hotel-name with value: super 5
                # [Info] Adding Slot: attraction-name with value: archway house
                # [Info] Adding Slot: train-arriveby with value: 8
                # [Info] Adding Slot: train-leaveat with value: 10
                # [Info] Adding Slot: restaurant-book time with value: 9
                # [Info] Adding Slot: hotel-name with value: nothamilton lodge
                # [Info] Adding Slot: attraction-name with value: st christs college

    return label_dict



def clean_original_ontology(values):
    new_values = set()
    
    for v in values:
        v = v.replace("guesthouse", "guest house").replace("theater", "theatre").replace("  ", " ")
        
        if v == "queens' college": v = "queen's college"
        if v == "gonvile hotel": v = "the gonvile hotel"
        if v == "worth house": v = "the worth house"
        if v == "acorn guest house": v = "the acorn guest house"
        if v == "golden house": v = "the golden house"    
        if v == "castle galleries": v = "the castle galleries"
        if v == "place": v = "the place"
        if v == "cambridge artworks": v = "the cambridge artworks"
        if v == "cambridge corn exchange": v = "the cambridge corn exchange"
        if v == "cambridge arts theatre": v = "the cambridge arts theatre"
        if v == "cambridge arts theater": v = "the cambridge arts theatre" 
        if v == "gonville hotel": v = "the gonville hotel" 
        if v == "dojo noodle bar": v = "the dojo noodle bar"
        if v == "alexander bed and breakfast": v = "the alexander bed and breakfast"
        if v == "allenbell": v = "the allenbell"
        if v == "galleria": v = "the galleria"
            
        
        if v.split(" ")[0] == "the" and " ".join(v.split(" ")[1:]) in new_values:
            new_values.remove(" ".join(v.split(" ")[1:]))
        new_values.add(v)
     
    new_values = list(new_values)
    return new_values
    
    