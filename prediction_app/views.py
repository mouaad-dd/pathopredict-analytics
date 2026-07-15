from django.shortcuts import render, redirect
from django.contrib import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import joblib
import numpy as np
import pandas as pd
import os
from django.conf import settings

BASE_DIR = settings.BASE_DIR
modele = joblib.load(os.path.join(BASE_DIR, 'modele_prediction_maladies.pkl'))
liste_symptomes = list(joblib.load(os.path.join(BASE_DIR, 'liste_symptomes.pkl')))

NOM_FRANCAIS_MALADIES = {
    "Drug Reaction": "Drug Reaction",
    "Malaria": "Malaria",
    "Allergy": "Allergy",
    "Diabetes": "Diabetes",
    "Dengue": "Dengue",
    "Typhoid": "Typhoid",
    "Fungal infection": "Fungal Infection",
    "Common Cold": "Common Cold",
    "Pneumonia": "Pneumonia",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids",
    "Heart attack": "Heart Attack",
    "Varicose veins": "Varicose Veins",
    "Hypothyroidism": "Hypothyroidism",
    "Hyperthyroidism": "Hyperthyroidism",
    "Hypoglycemia": "Hypoglycemia",
    "Osteoarthristis": "Osteoarthritis",
    "Arthritis": "Arthritis",
    "(vertigo) Paroymsal Positional Vertigo": "Paroxysmal Positional Vertigo",
    "Acne": "Acne",
    "Urinary tract infection": "Urinary Tract Infection",
    "Psoriasis": "Psoriasis",
    "Impetigo": "Impetigo",
    "Gastroenteritis": "Gastroenteritis",
    "Bronchial Asthma": "Bronchial Asthma",
    "Hypertension": "Hypertension",
    "Migraine": "Migraine",
    "Cervical spondylosis": "Cervical Spondylosis",
    "Paralysis (brain hemorrhage)": "Paralysis (Brain Hemorrhage)",
    "Jaundice": "Jaundice",
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
    "AIDS": "AIDS (HIV)",
    "Gerd": "Acid Reflux (GERD)",
    "Chronic cholestasis": "Chronic Cholestasis",
    "Pure hypocholesterolemia": "Pure Hypocholesterolemia",
    "Hepatitis A": "Hepatitis A",
    "Hepatitis B": "Hepatitis B",
    "Hepatitis C": "Hepatitis C",
    "Hepatitis D": "Hepatitis D",
    "Hepatitis E": "Hepatitis E",
    "Alcoholic hepatitis": "Alcoholic Hepatitis",
    "Tuberculosis": "Tuberculosis"
}

DESCRIPTIONS_MALADIES = {
    "Drug Reaction": "An adverse immune response of the body to a medication, typically causing skin rashes, itching, or fever.",
    "Malaria": "A parasitic infection transmitted by mosquitoes, leading to high fevers, shaking chills, headaches, and intense sweating.",
    "Allergy": "An exaggerated immune system reaction to a foreign substance (such as dust, pollen, or food), causing sneezing, itching, or redness.",
    "Diabetes": "A chronic metabolic disorder characterized by high blood sugar levels resulting from insufficient insulin production or ineffective insulin action.",
    "Dengue": "A mosquito-borne viral infection causing severe high fever, intense joint and muscle pain, headaches, and skin rashes.",
    "Typhoid": "A serious bacterial infection contracted by ingesting contaminated food or water, causing continuous high fever, abdominal pain, and fatigue.",
    "Fungal infection": "A localized skin or nail infection caused by microscopic fungi, leading to itching, redness, and peeling skin.",
    "Common Cold": "A highly contagious viral infection of the upper respiratory tract, resulting in runny nose, sneezing, sore throat, and mild fatigue.",
    "Pneumonia": "An acute infection of one or both lungs, usually bacterial or viral, causing a persistent cough, fever, difficulty breathing, and chest pain.",
    "Dimorphic hemmorhoids(piles)": "Swollen and painful veins in the anus and lower rectum, which can cause minor bleeding or pain during bowel movements.",
    "Heart attack": "A medical emergency caused by blocked blood flow to the heart muscle, leading to intense chest pain radiating to the arm or jaw.",
    "Varicose veins": "Gnarled, enlarged veins, most commonly appearing in the legs, caused by weak or damaged vein walls and valves.",
    "Hypothyroidism": "Underactive thyroid gland producing insufficient hormones, slowing down the metabolism and causing fatigue, weight gain, and cold sensitivity.",
    "Hyperthyroidism": "Overactive thyroid gland producing excessive hormones, accelerating the body's metabolism and causing rapid weight loss, palpitations, and anxiety.",
    "Hypoglycemia": "An abnormally low blood glucose level, manifesting as shakiness, cold sweats, dizziness, and sudden hunger.",
    "Osteoarthristis": "The most common form of arthritis, caused by the gradual wear and tear of joint cartilage, leading to pain and stiffness.",
    "Arthritis": "Inflammation of one or more joints, characterized by pain, swelling, warmth, and restricted movement.",
    "(vertigo) Paroymsal Positional Vertigo": "An inner ear disorder triggering short, intense episodes of spinning sensations triggered by specific head movements.",
    "Acne": "A common skin condition occurring when hair follicles become clogged with oil and dead skin cells, causing pimples and blackheads.",
    "Urinary tract infection": "A bacterial infection affecting any part of the urinary system, resulting in painful, burning urination and frequent urges.",
    "Psoriasis": "A chronic autoimmune skin disease that speeds up the life cycle of skin cells, causing thick, red patches covered with silvery scales.",
    "Impetigo": "A highly contagious bacterial skin infection, common in children, characterized by red sores that break open and form honey-colored crusts.",
    "Gastroenteritis": "Inflammation of the stomach and intestines due to infection, leading to diarrhea, vomiting, and abdominal cramps.",
    "Bronchial Asthma": "A chronic disease causing inflammation and narrowing of the airways, resulting in wheezing, shortness of breath, and coughing.",
    "Hypertension": "Chronically elevated blood pressure in the arteries, increasing the heart's workload over time without immediate warning symptoms.",
    "Migraine": "An intense, throbbing headache, typically affecting one side of the head, often accompanied by nausea and extreme sensitivity to light.",
    "Cervical spondylosis": "Age-related wear and tear affecting the spinal discs in the neck, leading to persistent neck pain and stiffness.",
    "Paralysis (brain hemorrhage)": "Loss of motor function caused by bleeding within the brain tissue, requiring immediate emergency medical intervention.",
    "Jaundice": "A yellowish discoloration of the skin and eyes caused by an excess of bilirubin in the blood, reflecting liver or gallbladder dysfunction.",
    "Peptic ulcer diseae": "Sores that develop on the inner lining of the stomach and the upper part of your small intestine, causing burning pain after meals.",
    "AIDS": "An advanced stage of HIV infection characterized by a severely compromised immune system, making the body vulnerable to opportunistic infections.",
    "Gerd": "Gastroesophageal reflux disease, where stomach acid flows back into the food pipe, causing heartburn and dry throat irritation.",
    "Chronic cholestasis": "A prolonged reduction or stoppage of bile flow, causing severe itching, fatigue, and sometimes jaundice.",
    "Pure hypocholesterolemia": "An abnormally low level of cholesterol in the blood, often associated with specific metabolic or nutritional factors.",
    "Hepatitis A": "An acute viral liver disease spread through contaminated food or water, resulting in fatigue, nausea, and loss of appetite.",
    "Hepatitis B": "A viral liver infection transmitted through blood or bodily fluids, potentially leading to chronic liver damage or scarring.",
    "Hepatitis C": "A blood-borne viral infection of the liver, frequently progressing to silent chronic inflammation and long-term liver issues.",
    "Hepatitis D": "A serious liver disease occurring only in people who are already infected with Hepatitis B, leading to accelerated liver complications.",
    "Hepatitis E": "A viral liver infection mainly transmitted through contaminated drinking water, usually self-limiting but dangerous to vulnerable profiles.",
    "Alcoholic hepatitis": "Acute or chronic liver inflammation caused by heavy, long-term toxic consumption of alcohol.",
    "Tuberculosis": "A highly contagious bacterial infection primarily attacking the lungs, characterized by persistent coughing, chest pain, and night sweats."
}

# Targeted recovery / first aid actions for each predicted disease
RECOMMENDATIONS_MALADIES = {
    "Drug Reaction": [
        "Immediately stop taking the suspected medication.",
        "Seek urgent medical evaluation from a physician.",
        "Take antihistamines if advised by a doctor to soothe mild itching.",
        "Monitor for severe signs like breathing difficulties or swelling (call emergency services if these occur)."
    ],
    "Malaria": [
        "Seek immediate clinical diagnostic testing (blood smear/RDT).",
        "Start prescribed antimalarial treatment without delay to prevent severe complications.",
        "Rest and drink plenty of fluids to stay hydrated.",
        "Use paracetamol to control high fevers as directed."
    ],
    "Allergy": [
        "Identify and avoid exposure to the triggering allergen.",
        "Use over-the-counter antihistamines or nasal sprays to manage symptoms.",
        "Keep an epinephrine auto-injector (EpiPen) nearby if you have a history of severe anaphylaxis.",
        "Consult an allergist if symptoms persist or worsen."
    ],
    "Diabetes": [
        "Monitor your blood glucose levels regularly.",
        "Follow your prescribed insulin or oral medication regimen.",
        "Maintain a structured, low-glycemic index diet and stay active.",
        "Carry fast-acting carbohydrates (like juice or glucose tablets) to treat sudden hypoglycemia."
    ],
    "Dengue": [
        "Rest as much as possible and stay highly hydrated.",
        "Take paracetamol to manage fever and pain; avoid aspirin or ibuprofen as they can increase bleeding risks.",
        "Prevent mosquito bites to stop the virus from spreading.",
        "Watch closely for warning signs of severe Dengue, such as persistent vomiting or bleeding."
    ],
    "Typhoid": [
        "Seek immediate medical attention to obtain a prescription for antibiotic therapy.",
        "Drink only clean, safe, bottled or boiled water.",
        "Maintain strict hand hygiene with soap and water.",
        "Eat fully cooked, hot meals and avoid raw foods."
    ],
    "Fungal infection": [
        "Apply over-the-counter or prescribed topical antifungal creams.",
        "Keep the affected skin clean, cool, and thoroughly dry.",
        "Avoid sharing personal items like towels, clothes, or hairbrushes.",
        "Wear loose-fitting, breathable cotton clothing."
    ],
    "Common Cold": [
        "Get plenty of rest to allow your body to recover.",
        "Stay hydrated by drinking plenty of water, broth, or hot teas.",
        "Use saline nasal sprays or room humidifiers to soothe congestion.",
        "Take paracetamol or ibuprofen to alleviate body aches and fever."
    ],
    "Pneumonia": [
        "Consult a doctor immediately to receive targeted antibiotics or antivirals.",
        "Get plenty of bed rest and limit physical exertion.",
        "Drink warm fluids to help loosen mucus in your chest.",
        "Take fever reducers if necessary and seek emergency care if you experience severe shortness of breath."
    ],
    "Dimorphic hemmorhoids(piles)": [
        "Eat a high-fiber diet (fruits, vegetables, whole grains) and drink plenty of water.",
        "Avoid straining during bowel movements.",
        "Take warm sitz baths for 15-20 minutes, 2 to 3 times a day.",
        "Use over-the-counter topical ointments or pain relievers if discomfort persists."
    ],
    "Heart attack": [
        "Call emergency services immediately—this is a life-threatening emergency.",
        "Chew and swallow an aspirin if recommended by emergency dispatch.",
        "Rest quietly while waiting for emergency responders; do not try to drive yourself to the hospital.",
        "Stay calm and sit down to ease pressure on the heart."
    ],
    "Varicose veins": [
        "Avoid standing or sitting for long periods without moving.",
        "Elevate your legs above heart level when resting.",
        "Wear graduated compression stockings daily.",
        "Exercise regularly to improve blood circulation in your legs."
    ],
    "Hypothyroidism": [
        "Take your prescribed daily thyroid hormone replacement medication (levothyroxine) on an empty stomach.",
        "Schedule regular blood tests to monitor your thyroid-stimulating hormone (TSH) levels.",
        "Consult your physician before taking new supplements that can interfere with absorption.",
        "Maintain a balanced diet to manage energy levels and metabolic changes."
    ],
    "Hyperthyroidism": [
        "Follow your prescribed antithyroid medication, beta-blockers, or treatment plan closely.",
        "Monitor your heart rate regularly and report palpitations to your doctor.",
        "Ensure adequate intake of calcium and Vitamin D to protect bone density.",
        "Limit excessive caffeine and stimulant intake to manage anxiety and rapid heart rate."
    ],
    "Hypoglycemia": [
        "Consume 15-20 grams of fast-acting glucose (e.g., fruit juice, candy, or glucose tablets) immediately.",
        "Recheck blood sugar levels after 15 minutes and repeat if levels remain low.",
        "Eat a small meal or snack containing complex carbohydrates once your blood sugar returns to normal.",
        "Notify your doctor to adjust your diabetes medication if hypoglycemia occurs frequently."
    ],
    "Osteoarthristis": [
        "Engage in regular, low-impact exercises (swimming, cycling, walking) to strengthen supporting muscles.",
        "Maintain a healthy weight to reduce pressure on weight-bearing joints.",
        "Apply hot packs to relieve stiffness or cold packs to reduce joint inflammation.",
        "Use over-the-counter pain relievers or topical gels as directed."
    ],
    "Arthritis": [
        "Consult a rheumatologist for a tailored treatment plan.",
        "Incorporate anti-inflammatory foods into your diet.",
        "Balance active exercises with adequate rest during flare-ups.",
        "Utilize joint protection techniques and ergonomic tools to reduce daily joint strain."
    ],
    "(vertigo) Paroymsal Positional Vertigo": [
        "Avoid sudden head movements and tilting your head far backward.",
        "Consult a physical therapist or specialist to perform the Epley maneuver.",
        "Sit down immediately when you feel a dizzy spell coming on.",
        "Ensure your home environment is safe from tripping hazards."
    ],
    "Acne": [
        "Wash your face gently twice a day with a mild, non-drying cleanser.",
        "Avoid squeezing, popping, or picking at pimples to prevent scarring.",
        "Use non-comedogenic (pore-clogging-free) skincare and cosmetic products.",
        "Apply targeted topical treatments containing salicylic acid or benzoyl peroxide."
    ],
    "Urinary tract infection": [
        "Drink plenty of water to help flush bacteria out of your urinary system.",
        "Consult a healthcare professional to obtain a prescription for antibiotic therapy.",
        "Avoid irritating fluids like alcohol, caffeine, and highly acidic drinks.",
        "Urinate promptly when you feel the urge and always wipe from front to back."
    ],
    "Psoriasis": [
        "Keep your skin well-moisturized with thick ointments or creams.",
        "Avoid dry or cold weather triggers if possible.",
        "Follow your prescribed topical steroids, phototherapy, or systemic treatments.",
        "Identify and manage stressors, which are common triggers for psoriasis flare-ups."
    ],
    "Impetigo": [
        "Consult a doctor for prescription topical or oral antibiotic ointment.",
        "Gently wash the sores with soap and warm water to remove loose crusts.",
        "Cover the sores with a clean bandage to prevent spreading the infection.",
        "Keep your nails short and avoid scratching the affected areas."
    ],
    "Gastroenteritis": [
        "Focus on replacing lost fluids with oral rehydration solutions, clear broths, or water.",
        "Eat small, bland meals (such as bananas, rice, applesauce, toast) once vomiting stops.",
        "Avoid dairy, caffeine, alcohol, fatty, and highly seasoned foods.",
        "Maintain strict handwashing habits to protect others from getting infected."
    ],
    "Bronchial Asthma": [
        "Keep your quick-relief (rescue) inhaler with you at all times.",
        "Identify and minimize exposure to asthma triggers (pollen, dust mites, cold air, smoke).",
        "Follow your personalized asthma action plan and take long-term controller medications regularly.",
        "Seek immediate emergency care if your breathing does not improve after using your rescue inhaler."
    ],
    "Hypertension": [
        "Maintain a low-sodium, heart-healthy diet.",
        "Engage in moderate aerobic physical activity for at least 30 minutes most days.",
        "Monitor your blood pressure levels at home and keep a log for your doctor.",
        "Take your prescribed antihypertensive medications consistently."
    ],
    "Migraine": [
        "Rest in a quiet, dark, and cool room at the onset of symptoms.",
        "Apply a cold compress to your forehead or the back of your neck.",
        "Hydrate with small sips of water and avoid triggering foods.",
        "Take abortive migraine medications as directed by your physician."
    ],
    "Cervical spondylosis": [
        "Practice good posture while sitting, standing, and working at a computer.",
        "Perform regular gentle neck stretching and strengthening exercises.",
        "Apply a heating pad or ice pack to relieve localized neck pain.",
        "Use a supportive neck pillow designed for cervical alignment while sleeping."
    ],
    "Paralysis (brain hemorrhage)": [
        "Call emergency medical services immediately—this is a critical emergency.",
        "Do not give the person anything to eat or drink.",
        "Help the person lie down on their side with their head slightly elevated if they are conscious.",
        "Monitor their breathing and responsiveness closely until emergency personnel arrive."
    ],
    "Jaundice": [
        "Consult a physician promptly to diagnose and treat the underlying hepatic or biliary issue.",
        "Avoid all alcohol and medications that can strain or damage the liver.",
        "Stay hydrated and consume a nutritious, easily digestible diet.",
        "Monitor for severe signs like abdominal swelling, confusion, or severe bleeding."
    ],
    "Peptic ulcer diseae": [
        "Follow your prescribed regimen, including proton pump inhibitors (PPIs) or antibiotics.",
        "Avoid nonsteroidal anti-inflammatory drugs (NSAIDs) like aspirin or ibuprofen.",
        "Limit foods that irritate your stomach, such as spicy, fatty, or highly acidic items.",
        "Manage stress levels and avoid smoking or alcohol consumption."
    ],
    "AIDS": [
        "Consult an infectious disease specialist for antiretroviral therapy (ART) management.",
        "Adhere strictly to your daily medication schedule to maintain viral suppression.",
        "Protect yourself from opportunistic infections by practicing food safety and maintaining up-to-date vaccinations.",
        "Schedule regular clinical follow-ups to track your CD4 cell count and viral load."
    ],
    "Gerd": [
        "Eat smaller, more frequent meals instead of large, heavy portions.",
        "Avoid lying down for at least 3 hours after eating.",
        "Elevate the head of your bed by 6 to 9 inches.",
        "Limit trigger foods, including fried dishes, chocolate, mint, caffeine, and acidic items."
    ],
    "Chronic cholestasis": [
        "Consult a hepatologist for targeted diagnostic and therapeutic management.",
        "Follow a low-fat diet and consider fat-soluble vitamin supplements (A, D, E, K) under doctor supervision.",
        "Use prescribed medications to alleviate severe itching (pruritus).",
        "Monitor liver function tests regularly."
    ],
    "Pure hypocholesterolemia": [
        "Consult your healthcare provider to investigate underlying causes, such as malabsorption or hyperthyroidism.",
        "Adopt a balanced, nutrient-dense diet rich in healthy fats.",
        "Monitor your nutritional status regularly.",
        "Discuss any necessary adjustments to your diet or medications with your physician."
    ],
    "Hepatitis A": [
        "Ensure plenty of bed rest and avoid strenuous physical activity.",
        "Stay hydrated by sipping water, broth, or electrolyte drinks.",
        "Avoid all alcohol and consult your doctor before taking any over-the-counter medications to protect your liver.",
        "Wash hands thoroughly to prevent spreading the virus to household members."
    ],
    "Hepatitis B": [
        "Consult a specialist to determine if antiviral medications are necessary.",
        "Avoid alcohol and liver-toxic substances.",
        "Eat a balanced, healthy diet and get adequate rest.",
        "Take precautions to prevent transmitting the virus to partners or family members."
    ],
    "Hepatitis C": [
        "Consult a gastroenterologist or hepatologist to discuss direct-acting antiviral (DAA) treatment.",
        "Avoid alcohol and check with your physician before starting any new supplements or medications.",
        "Cover cuts and sores, and do not share personal items like razors or toothbrushes.",
        "Monitor liver health indicators through regular clinical check-ups."
    ],
    "Hepatitis D": [
        "Consult a liver disease specialist for specialized evaluation and monitoring.",
        "Adhere to treatment strategies targeting both Hepatitis B and D co-infections.",
        "Avoid alcohol and eat a nutrient-rich diet to support liver function.",
        "Get regular blood tests to track viral activity and liver health."
    ],
    "Hepatitis E": [
        "Get plenty of physical rest and stay well-hydrated.",
        "Avoid alcohol and unnecessary over-the-counter drugs.",
        "Seek immediate medical attention if you are pregnant, as Hepatitis E can lead to severe complications.",
        "Ensure all drinking water is clean and food is thoroughly cooked."
    ],
    "Alcoholic hepatitis": [
        "Stop consuming alcohol entirely and permanently.",
        "Consult a healthcare professional for specialized medical and nutritional support.",
        "Follow a high-protein, calorie-dense diet to correct nutritional deficiencies.",
        "Monitor closely for severe liver complications like jaundice or fluid buildup."
    ],
    "Tuberculosis": [
        "Consult a specialist and strictly follow your prescribed multi-drug treatment course.",
        "Complete the entire antibiotic regimen to prevent drug-resistant strains.",
        "Stay in a well-ventilated room and wear a mask around others during the initial infectious phase.",
        "Maintain a balanced diet and get plenty of rest to support immune recovery."
    ]
}

@login_required
def index(request):
    resultat = None
    description = None
    symptomes_coches = []
    top_predictions = []
    confiance_principale = 0  # Default value for GET requests
    recommendations = []      # Default empty list
    error_message = None      # Add an error tracker variable

    if request.method == 'POST':
        symptomes_coches = request.POST.getlist('symptomes')
        
        # -------------------------------------------------------------
        # FIX: Check if no symptoms are selected
        # -------------------------------------------------------------
        if not symptomes_coches:
            error_message = "Please select at least one clinical symptom before executing the statistical analysis."
        else:
            # Continue standard ML prediction logic if at least one checkbox is checked
            vecteur_input = np.zeros(len(liste_symptomes))
            
            for symp in symptomes_coches:
                if symp in liste_symptomes:
                    index_symp = liste_symptomes.index(symp)
                    vecteur_input[index_symp] = 1

            df_input = pd.DataFrame([vecteur_input], columns=liste_symptomes)

            probabilites = modele.predict_proba(df_input)[0]
            classes = modele.classes_
            
            predictions_associees = sorted(zip(classes, probabilites), key=lambda x: x[1], reverse=True)
            
            raw_resultat = predictions_associees[0][0]
            resultat = NOM_FRANCAIS_MALADIES.get(raw_resultat, raw_resultat)
            description = DESCRIPTIONS_MALADIES.get(raw_resultat, "No description available.")
            confiance_principale = int(predictions_associees[0][1] * 100)
            
            # Pull clinical recommendations based on the predicted target
            recommendations = RECOMMENDATIONS_MALADIES.get(raw_resultat, ["Consult a medical professional for personalized advice."])
            
            for mal, prob in predictions_associees[:3]:
                if prob > 0:
                    top_predictions.append({
                        'nom': NOM_FRANCAIS_MALADIES.get(mal, mal),
                        'score': int(prob * 100)
                    })

    context = {
        'liste_symptomes': liste_symptomes,
        'resultat': resultat,
        'description': description,
        'symptomes_coches': symptomes_coches,
        'top_predictions': top_predictions,
        'confiance_principale': confiance_principale,
        'recommendations': recommendations,
        'error_message': error_message,  # Pass the validation error to template context
    }
    return render(request, 'prediction_app/index.html', context)

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def intro_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    return render(request, 'prediction_app/intro.html')