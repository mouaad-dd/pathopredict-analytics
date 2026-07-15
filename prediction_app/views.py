from django.shortcuts import render, redirect
from django.contrib.auth import login
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
    "Drug Reaction": "Réaction indésirable aux médicaments",
    "Malaria": "Paludisme (Malaria)",
    "Allergy": "Allergie",
    "Diabetes": "Diabète",
    "Dengue": "Dengue",
    "Typhoid": "Fièvre typhoïde",
    "Fungal infection": "Infection fongique",
    "Common Cold": "Rhume banal",
    "Pneumonia": "Pneumonie",
    "Dimorphic hemmorhoids(piles)": "Hémorroïdes",
    "Heart attack": "Crise cardiaque",
    "Varicose veins": "Varices",
    "Hypothyroidism": "Hypothyroïdie",
    "Hyperthyroidism": "Hyperthyroïdie",
    "Hypoglycemia": "Hypoglycémie",
    "Osteoarthristis": "Arthrose (Ostéoarthrite)",
    "Arthritis": "Arthrite",
    "(vertigo) Paroymsal Positional Vertigo": "Vertige positionnel paroxystique bénin",
    "Acne": "Acné",
    "Urinary tract infection": "Infection urinaire",
    "Psoriasis": "Psoriasis",
    "Impetigo": "Impétigo",
    "Gastroenteritis": "Gastro-entérite",
    "Bronchial Asthma": "Asthme bronchique",
    "Hypertension": "Hypertension artérielle",
    "Migraine": "Migraine",
    "Cervical spondylosis": "Spondylarthrose cervicale",
    "Paralysis (brain hemorrhage)": "Paralysie (Hémorragie cérébrale)",
    "Jaundice": "Jaunisse (Ictère)",
    "Peptic ulcer diseae": "Ulcère peptique",
    "AIDS": "SIDA (VIH)",
    "Gerd": "Reflux gastro-œsophagien (RGO)",
    "Chronic cholestasis": "Cholestase chronique",
    "Pure hypocholesterolemia": "Hypocholestérolémie pure",
    "Hepatitis A": "Hépatite A",
    "Hepatitis B": "Hépatite B",
    "Hepatitis C": "Hépatite C",
    "Hepatitis D": "Hépatite D",
    "Hepatitis E": "Hépatite E",
    "Alcoholic hepatitis": "Hépatite alcoolique",
    "Tuberculosis": "Tuberculose"
}

TRADUCTION_SYMPTOMES = {
    "itching": "Démangeaisons", "skin_rash": "Éruption cutanée", "continuous_sneezing": "Éternuements continus",
    "shivering": "Frissons", "chills": "Courbatures / Frissons froids", "joint_pain": "Douleurs articulaires",
    "stomach_pain": "Maux d'estomac", "acidity": "Acidité gastrique", "ulcers_on_tongue": "Aphtes sur la langue",
    "muscle_wasting": "Fonte musculaire", "vomiting": "Vomissements", "burning_micturition": "Brûlure mictionnelle (en urinant)",
    "spotting_urination": "Traces de sang dans les urines", "fatigue": "Fatigue", "weight_gain": "Prise de poids",
    "anxiety": "Anxiété", "cold_hands_and_feets": "Mains et pieds froids", "mood_swings": "Changements d'humeur",
    "weight_loss": "Perte de poids", "restlessness": "Agitation", "lethargy": "Léthargie", "patches_in_throat": "Plaques dans la gorge",
    "irregular_sugar_level": "Taux de sucre irrégulier", "cough": "Toux", "high_fever": "Fièvre élevée",
    "sunken_eyes": "Yeux creux", "breathlessness": "Essoufflement", "sweating": "Transpiration excessive",
    "dehydration": "Déshydratation", "indigestion": "Indigestions", "headache": "Maux de tête", "yellowish_skin": "Peau jaunâtre",
    "dark_urine": "Urines foncées", "nausea": "Nausées", "loss_of_appetite": "Perte d'appétit", "pain_behind_the_eyes": "Douleur derrière les yeux",
    "back_pain": "Mal de dos", "constipation": "Constipation", "abdominal_pain": "Douleur abdominale", "diarrhoea": "Diarrhée",
    "mild_fever": "Fièvre légère", "yellow_urine": "Urines jaunes", "yellowing_of_eyes": "Jaunissement des yeux",
    "acute_liver_failure": "Insuffisance hépatique aiguë", "fluid_overload": "Surcharge de fluides", "swelling_of_stomach": "Gonflement de l'estomac",
    "swelled_lymph_nodes": "Ganglions lymphatiques gonflés", "malaise": "Malaise général", "blurred_and_distorted_vision": "Vision floue et déformée",
    "phlegm": "Crachats / Flegme", "throat_irritation": "Irritation de la gorge", "redness_of_eyes": "Rougeur des yeux",
    "sinus_pressure": "Pression sinusale", "runny_nose": "Écoulement nasal", "congestion": "Congestion nasale",
    "chest_pain": "Douleur thoracique", "weakness_in_limbs": "Faiblesse des membres", "fast_heart_rate": "Rythme cardiaque rapide",
    "pain_during_bowel_movements": "Douleur lors de la défécation", "pain_in_anal_region": "Douleur dans la région anale",
    "bloody_stool": "Selles sanglantes", "irritation_in_anus": "Irritation de l'anus", "neck_pain": "Douleur au cou",
    "dizziness": "Sensations de vertige", "cramps": "Crampes", "bruising": "Ecchymoses / Bleus", "obesity": "Obésité",
    "swollen_legs": "Jambes enflées", "swollen_blood_vessels": "Vaisseaux sanguins gonflés", "puffy_face_and_eyes": "Visage et yeux bouffis",
    "enlarged_thyroid": "Gorge / Thyroïde gonflée", "brittle_nails": "Ongles cassants", "swollen_extremeties": "Extrémités gonflées",
    "excessive_hunger": "Faim excessive", "extra_marital_contacts": "Rapports sexuels non protégés", "drying_and_tingling_lips": "Lèvres sèches et picotements",
    "slurred_speech": "Troubles de la parole", "knee_pain": "Douleur au genou", "hip_joint_pain": "Douleur à la hanche",
    "muscle_weakness": "Faiblesse musculaire", "stiff_neck": "Raideur de la nuque", "swelling_joints": "Gonflement des articulations",
    "movement_stiffness": "Rigidité des mouvements", "spinning_movements": "Sensations de rotation", "loss_of_balance": "Perte d'équilibre",
    "unsteadiness": "Instabilité physique", "weakness_of_one_body_side": "Faiblesse d'un côté du corps", "loss_of_smell": "Perte d'odorat",
    "bladder_discomfort": "Inconfort vésical", "foul_smell_of_urine": "Urine malodorante", "continuous_feel_of_urine": "Envie constante d'uriner",
    "passage_of_gases": "Passage de gaz", "internal_itching": "Démangeaisons internes", "toxic_look_(typhos)": "Faciès toxique",
    "depression": "Dépression", "irritability": "Irritabilité", "muscle_pain": "Douleurs musculaires", "altered_sensorium": "Altération de la conscience",
    "red_spots_over_body": "Taches rouges sur le corps", "belly_pain": "Maux de ventre", "abnormal_menstruation": "Règles anormales",
    "dischromic_patches": "Plaques décolorées", "watering_from_eyes": "Yeux larmoyants", "increased_appetite": "Augmentation de l'appétit",
    "polyuria": "Miction excessive (Polyurie)", "family_history": "Antécédents patrimoniaux", "mucoid_sputum": "Crachats muqueux",
    "rusty_sputum": "Crachats rouillés", "lack_of_concentration": "Manque de concentration", "visual_disturbances": "Troubles visuels",
    "receiving_blood_transfusion": "Historique de transfusion sanguine", "receiving_unsterile_injections": "Injections non stériles",
    "coma": "Coma", "stomach_bleeding": "Saignement de l'estomac", "distention_of_abdomen": "Distension de l'abdomen",
    "history_of_alcohol_consumption": "Antécédents de consommation d'alcool", "blood_in_sputum": "Sang dans les crachats",
    "prominent_veins_on_calf": "Veines saillantes sur le mollet", "palpitations": "Palpitations", "painful_walking": "Marche douloureuse",
    "pus_filled_pimples": "Boutons de pus", "blackheads": "Points noirs", "scurring": "Cicatrices d'acné", "skin_peeling": "Desquamation de la peau",
    "silver_like_dusting": "Pellicules argentées (psoriasis)", "small_dents_in_nails": "Petits creux sur les ongles",
    "inflammatory_nails": "Inflammation des ongles", "blister": "Ampoules / Cloches", "red_sore_around_nose": "Plaies rouges autour du nez",
    "yellow_crust_ooze": "Suintement de croûtes jaunes"
}

DESCRIPTIONS_MALADIES_FR = {
    "Drug Reaction": "Une réaction immunitaire indésirable de l'organisme à un médicament, provoquant généralement des éruptions cutanées, des démangeaisons ou de la fièvre.",
    "Malaria": "Une infection parasitaire transmise par les moustiques, entraînant de fortes fièvres, des frissons, des maux de tête et une transpiration intense.",
    "Allergy": "Une réaction exagérée du système immunitaire à une substance étrangère (poussière, pollen, aliments), provoquant éternuements, démangeaisons ou rougeurs.",
    "Diabetes": "Un trouble métabolique chronique caractérisé par un taux de sucre élevé dans le sang, résultant d'une production insuffisante d'insuline.",
    "Dengue": "Une infection virale transmise par les moustiques, provoquant une forte fièvre, d'intenses douleurs musculaires et des éruptions cutanées.",
    "Typhoid": "Une infection bactérienne grave contractée par l'ingestion d'eau ou d'aliments contaminés, provoquant une fièvre continue et des douleurs abdominales.",
    "Fungal infection": "Une infection cutanée ou unguéale localisée causée par des champignons microscopiques, entraînant démangeaisons, rougeurs et desquamation.",
    "Common Cold": "Une infection virale hautement contagieuse des voies respiratoires supérieures, entraînant un écoulement nasal, des éternuements et des maux de gorge.",
    "Pneumonia": "Une infection aiguë d'un ou des deux poumons, généralement bactérienne ou virale, provoquant une toux persistante, de la fièvre et des difficultés respiratoires.",
    "Dimorphic hemmorhoids(piles)": "Veines gonflées et douloureuses dans l'anus et le rectum inférieur, pouvant provoquer des saignements mineurs ou des douleurs lors de la défécation.",
    "Heart attack": "Une urgence médicale causée par l'interruption du flux sanguin vers le muscle cardiaque, provoquant une vive douleur thoracique irradiant vers le bras ou la mâchoire.",
    "Varicose veins": "Veines dilatées et tortueuses, apparaissant le plus souvent sur les jambes, dues à une faiblesse des parois veineuses.",
    "Hypothyroidism": "Insuffisance d'activité de la glande thyroïde, ralentissant le métabolisme et entraînant fatigue, prise de poids et sensibilité au froid.",
    "Hyperthyroidism": "Hyperactivité de la glande thyroïde entraînant une production excessive d'hormones, accélérant le métabolisme (perte de poids, palpitations).",
    "Hypoglycemia": "Une baisse anormale du taux de glucose dans le sang, se manifestant par des tremblements, des sueurs froides, des vertiges et une faim soudaine.",
    "Osteoarthristis": "La forme la plus courante d'arthrite, causée par l'usure progressive du cartilage articulaire, entraînant douleur et raideur.",
    "Arthritis": "Inflammation d'une ou plusieurs articulations, caractérisée par des douleurs, des gonflements, de la chaleur et une limitation des mouvements.",
    "(vertigo) Paroymsal Positional Vertigo": "Un trouble de l'oreille interne provoquant de brefs épisodes de vertiges intenses lors de mouvements de tête spécifiques.",
    "Acne": "Une affection cutanée courante survenant lorsque les follicules pileux sont obstrués par du sébum et des cellules mortes, causant des boutons.",
    "Urinary tract infection": "Une infection bactérienne affectant une partie du système urinaire, entraînant des mictions douloureuses, des brûlures et des envies fréquentes.",
    "Psoriasis": "Une maladie auto-immune cutanée chronique qui accélère le renouvellement cellulaire, provoquant des plaques rouges épaisses recouvertes de squames.",
    "Impetigo": "Une infection bactérienne cutanée très contagieuse, fréquente chez l'enfant, caractérisée par des lésions rouges formant des croûtes dorées.",
    "Gastroenteritis": "Inflammation de l'estomac et des intestins due à une infection, provoquant des diarrhées, des vomissements et des crampes abdominales.",
    "Bronchial Asthma": "Une maladie chronique provoquant l'inflammation et le rétrécissement des voies respiratoires, entraînant une respiration sifflante et un essoufflement.",
    "Hypertension": "Élévation chronique de la pression artérielle dans les artères, augmentant à long terme la charge de travail du cœur.",
    "Migraine": "Un mal de tête intense et lancinant, affectant généralement un seul côté de la tête, souvent accompagné de nausées et d'une sensibilité à la lumière.",
    "Cervical spondylosis": "Usure liée à l'âge affectant les disques intervertébraux du cou, entraînant des douleurs persistantes et une raideur de la nuque.",
    "Paralysis (brain hemorrhage)": "Perte de la fonction motrice causée par un saignement dans le tissu cérébral, nécessitant une prise en charge médicale d'urgence.",
    "Jaundice": "Une coloration jaunâtre de la peau et des yeux causée par un excès de bilirubine dans le sang, reflétant un dysfonctionnement du foie.",
    "Peptic ulcer diseae": "Lésions qui se développent sur la paroi interne de l'estomac ou de l'intestin grêle, provoquant des brûlures d'estomac après les repas.",
    "AIDS": "Stade avancé de l'infection par le VIH caractérisé par un système immunitaire gravement affaibli, rendant l'organisme vulnérable aux infections opportunistes.",
    "Gerd": "Reflux gastro-œsophagien, où l'acide gastrique remonte dans l'œsophage, provoquant des brûlures d'estomac et des irritations de la gorge.",
    "Chronic cholestasis": "Diminution ou arrêt prolongé de l'écoulement de la bile, provoquant des démangeaisons intenses, de la fatigue et parfois une jaunisse.",
    "Pure hypocholesterolemia": "Un taux anormalement bas de cholestérol dans le sang, souvent associé à des facteurs métaboliques ou nutritionnels spécifiques.",
    "Hepatitis A": "Une maladie hépatique virale aiguë transmise par de l'eau ou des aliments contaminés, entraînant fatigue, nausées et perte d'appétit.",
    "Hepatitis B": "Infection hépatique virale transmise par le sang ou les fluides corporels, pouvant entraîner des lésions hépatiques chroniques.",
    "Hepatitis C": "Infection virale du foie transmise par le sang, évoluant fréquemment vers une inflammation chronique silencieuse.",
    "Hepatitis D": "Maladie hépatique grave survenant uniquement chez les personnes déjà infectées par l'hépatite B, accélérant les complications.",
    "Hepatitis E": "Infection hépatique virale principalement transmise par de l'eau de boisson contaminée, généralement bénigne mais dangereuse pour les femmes enceintes.",
    "Alcoholic hepatitis": "Inflammation aiguë ou chronique du foie causée par une consommation excessive et prolongée d'alcool.",
    "Tuberculosis": "Une infection bactérienne hautement contagieuse attaquant principalement les poumons, caractérisée par une toux persistante et des sueurs nocturnes."
}

DESCRIPTIONS_MALADIES_EN = {
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

RECOMMENDATIONS_MALADIES_FR = {
    "Drug Reaction": ["Arrêtez immédiatement le médicament suspecté.", "Consultez d'urgence un médecin.", "Prenez des antihistaminiques si conseillé.", "Surveillez les signes respiratoires graves."],
    "Malaria": ["Consultez immédiatement pour un test de diagnostic rapide.", "Commencez sans délai le traitement antipaludique prescrit.", "Reposez-vous et buvez beaucoup d'eau.", "Utilisez du paracétamol pour contrôler la fièvre."],
    "Allergy": ["Évitez l'exposition à l'allergène déclencheur.", "Utilisez des antihistaminiques ou sprays nasaux.", "Gardez un EpiPen à portée de main si nécessaire.", "Consultez un allergologue si cela persiste."],
    "Diabetes": ["Surveillez régulièrement votre glycémie.", "Respectez scrupuleusement votre traitement prescrit.", "Maintenez une alimentation à faible indice glycémique.", "Ayez toujours du sucre rapide sur vous."],
    "Dengue": ["Reposez-vous et assurez-vous d'avoir une excellente hydratation.", "Prenez du paracétamol ; évitez l'aspirine ou l'ibuprofène.", "Protégez-vous des piqûres de moustiques.", "Surveillez l'apparition de signes d'alerte graves."],
    "Typhoid": ["Consultez immédiatement pour une antibiothérapie adaptée.", "Ne buvez que de l'eau propre (bouteille ou bouillie).", "Lavez-vous fréquemment les mains au savon.", "Consommez des repas chauds entièrement cuits."],
    "Fungal infection": ["Appliquez des pommades antifongiques locales.", "Gardez la zone touchée propre et sèche.", "Ne partagez pas vos effets personnels.", "Portez des vêtements amples en coton."],
    "Common Cold": ["Reposez-vous au maximum pour récupérer.", "Hydratez-vous en buvant de l'eau ou des tisanes.", "Utilisez des sprays nasaux salins.", "Prenez du paracétamol contre les courbatures."],
    "Pneumonia": ["Consultez d'urgence un médecin pour des antibiotiques.", "Restez au lit et limitez tout effort.", "Buvez des liquides chauds pour fluidifier le mucus.", "Appelez les urgences en cas de détresse respiratoire."],
    "Dimorphic hemmorhoids(piles)": ["Adoptez un régime très riche en fibres.", "Évitez les efforts de poussée excessifs.", "Prenez des bains de siège tièdes.", "Utilisez des crèmes locales apaisantes."],
    "Heart attack": ["Appelez immédiatement le SAMU (15) - Urgence vitale !", "Croquez de l'aspirine si validé par les secours.", "Restez immobile et assis en attendant.", "Essayez de respirer le plus calmement possible."],
    "Varicose veins": ["Évitez de rester debout de manière prolongée.", "Surélevez vos jambes au repos.", "Portez vos bas de contention au quotidien.", "Pratiquez une activité physique régulière."],
    "Hypothyroidism": ["Prenez votre traitement hormonal strictement à jeun.", "Contrôlez régulièrement vos taux de TSH.", "Consultez avant de prendre de nouveaux compléments.", "Adoptez un régime alimentaire équilibré."],
    "Hyperthyroidism": ["Prenez rigoureusement vos antithyroïdiens prescrits.", "Surveillez vos pulsations cardiaques.", "Assurez des apports suffisants en calcium.", "Limitez le café et les excitants."],
    "Hypoglycemia": ["Consommez immédiatement 15g de sucre rapide.", "Contrôlez la glycémie après 15 minutes.", "Prenez une collation de glucides complexes.", "Consultez pour adapter vos doses de traitement."],
    "Osteoarthristis": ["Pratiquez des exercices à faible impact.", "Maintenez un poids de forme pour soulager vos articulations.", "Appliquez du chaud (raideur) ou du froid (poussée).", "Utilisez des gels anti-inflammatoires locaux."],
    "Arthritis": ["Consultez un rhumatologue pour un suivi de fond.", "Incorporez des aliments anti-inflammatoires.", "Reposez-vous pendant les poussées douloureuses.", "Utilisez des aides ergonomiques au quotidien."],
    "(vertigo) Paroymsal Positional Vertigo": ["Évitez les mouvements brusques de la tête.", "Consultez pour réaliser une manœuvre d'Epley.", "Asseyez-vous immédiatement dès les premiers signes.", "Sécurisez votre domicile pour éviter les chutes."],
    "Acne": ["Nettoyez votre visage avec un gel doux non comédogène.", "Évitez absolument de percer vos boutons.", "Utilisez des cosmétiques non comédogènes.", "Appliquez des soins à l'acide salicylique."],
    "Urinary tract infection": ["Buvez d'importantes quantités d'eau.", "Consultez rapidement pour des antibiotiques.", "Évitez l'alcool, le café et le sucre.", "Urinez dès que l'envie se fait sentir."],
    "Psoriasis": ["Hydratez généreusement votre peau au quotidien.", "Évitez l'exposition prolongée au froid sec.", "Suivez scrupuleusement votre traitement prescrit.", "Gérez votre stress, puissant déclencheur."],
    "Impetigo": ["Consultez pour un traitement antibiotique local.", "Nettoyez à l'eau tiède pour retirer les croûtes.", "Couvrez les lésions d'une compresse stérile.", "Gardez des ongles courts et propres."],
    "Gastroenteritis": ["Privilégiez la réhydratation par petites gorgées.", "Reprenez une alimentation légère (riz, banane).", "Évitez les produits laitiers et les graisses.", "Lavez-vous minutieusement les mains au savon."],
    "Bronchial Asthma": ["Gardez en permanence votre inhalateur de secours.", "Limitez les facteurs déclencheurs (poussière, tabac).", "Prenez régulièrement votre traitement de fond.", "Appelez les urgences si la crise persiste."],
    "Hypertension": ["Adoptez une alimentation pauvre en sel.", "Pratiquez 30 minutes de marche par jour.", "Mesurez régulièrement votre tension artérielle.", "Prenez vos traitements tous les jours."],
    "Migraine": ["Allongez-vous dans une pièce sombre et silencieuse.", "Appliquez une compresse froide sur vos tempes.", "Buvez de l'eau et évitez les repas lourds.", "Prenez votre traitement spécifique dès le début."],
    "Cervical spondylosis": ["Veillez à maintenir une bonne posture assise.", "Réalisez de légers étirements de la nuque.", "Appliquez une source de chaleur douce.", "Utilisez un oreiller ergonomique adapté."],
    "Paralysis (brain hemorrhage)": ["Appelez immédiatement les secours médicaux !", "Ne donnez rien à manger ou à boire.", "Allongez la personne sur le côté si inconsciente.", "Surveillez en continu sa respiration."],
    "Jaundice": ["Consultez sans tarder un médecin pour le foie.", "Proscrivez totalement l'alcool et l'automédication.", "Mangez sainement avec des repas faciles à digérer.", "Signalez tout trouble de la conscience."],
    "Peptic ulcer diseae": ["Prenez votre traitement anti-acide prescrit.", "Évitez l'ibuprofène et l'aspirine sans avis.", "Limitez les plats très épicés ou acides.", "Supprimez le tabac et l'alcool."],
    "AIDS": ["Consultez régulièrement votre spécialiste TRV.", "Prenez vos médicaments chaque jour à heure fixe.", "Protégez-vous des infections opportunistes.", "Effectuez vos bilans sanguins régulièrement."],
    "Gerd": ["Prenez des repas plus légers et fractionnés.", "Attendez 3 heures avant de vous allonger.", "Surélevez la tête de votre lit.", "Évitez le chocolat, le café et les graisses."],
    "Chronic cholestasis": ["Consultez un hépatologue pour votre bilan.", "Adoptez un régime pauvre en graisses.", "Utilisez les traitements contre les démangeaisons.", "Surveillez régulièrement votre bilan hépatique."],
    "Pure hypocholesterolemia": ["Consultez votre médecin pour en chercher la cause.", "Incorporez de bonnes graisses d'origine végétale.", "Faites un suivi nutritionnel régulier.", "Adaptez votre prise en charge globale."],
    "Hepatitis A": ["Restez au repos complet sans effort intense.", "Hydratez-vous fréquemment.", "Proscrivez l'alcool et le paracétamol sans avis.", "Lavez-vous soigneusement les mains."],
    "Hepatitis B": ["Consultez un hépatologue pour le suivi.", "Évitez impérativement toutes les boissons alcoolisées.", "Consommez des repas équilibrés.", "Prenez des précautions lors des rapports."],
    "Hepatitis C": ["Consultez pour bénéficier des nouveaux antiviraux.", "Évitez l'alcool et l'automédication.", "Ne partagez aucun objet d'hygiène personnelle.", "Faites surveiller régulièrement l'état de votre foie."],
    "Hepatitis D": ["Consultez un spécialiste pour un suivi conjoint B/D.", "Respectez scrupuleusement le protocole.", "Adoptez un mode de vie sain sans alcool.", "Contrôlez régulièrement vos enzymes hépatiques."],
    "Hepatitis E": ["Assurez-vous un repos de qualité et buvez de l'eau.", "Évitez l'alcool et les médicaments non prescrits.", "Consultez d'urgence si vous êtes enceinte.", "Consommez de l'eau potable parfaitement sûre."],
    "Alcoholic hepatitis": ["Arrêtez définitivement toute consommation d'alcool.", "Faites-vous accompagner par un addictologue.", "Suivez un programme nutritionnel riche en protéines.", "Surveillez de près les complications."],
    "Tuberculosis": ["Prenez rigoureusement vos antibiotiques prescrits.", "Ne stoppez jamais votre traitement de fond.", "Restez isolé dans une pièce bien aérée au départ.", "Adoptez une alimentation fortifiante."]
}

RECOMMENDATIONS_MALADIES_EN = {
    "Drug Reaction": ["Immediately stop taking the suspected medication.", "Seek urgent medical evaluation from a physician.", "Take antihistamines if advised by a doctor.", "Monitor for severe signs like breathing difficulties."],
    "Malaria": ["Seek immediate clinical diagnostic testing.", "Start prescribed antimalarial treatment without delay.", "Rest and drink plenty of fluids.", "Use paracetamol to control high fevers."],
    "Allergy": ["Identify and avoid exposure to the triggering allergen.", "Use antihistamines or nasal sprays.", "Keep an epinephrine auto-injector nearby.", "Consult an allergist if symptoms persist."],
    "Diabetes": ["Monitor your blood glucose levels regularly.", "Follow your prescribed medication regimen.", "Maintain a structured, low-glycemic index diet.", "Carry fast-acting carbohydrates with you."],
    "Dengue": ["Rest as much as possible and stay hydrated.", "Take paracetamol; avoid aspirin or ibuprofen.", "Prevent mosquito bites.", "Watch closely for warning signs like vomiting."],
    "Typhoid": ["Seek immediate medical attention for antibiotics.", "Drink only clean, safe, bottled or boiled water.", "Maintain strict hand hygiene with soap.", "Eat fully cooked, hot meals."],
    "Fungal infection": ["Apply topical antifungal creams.", "Keep the affected skin clean and dry.", "Avoid sharing personal items like towels.", "Wear loose-fitting cotton clothing."],
    "Common Cold": ["Get plenty of rest to allow recovery.", "Stay hydrated by drinking water or hot teas.", "Use saline nasal sprays.", "Take paracetamol for body aches."],
    "Pneumonia": ["Consult a doctor immediately for targeted antibiotics.", "Get plenty of bed rest.", "Drink warm fluids to loosen mucus.", "Seek emergency care if breathing worsens."],
    "Dimorphic hemmorhoids(piles)": ["Eat a high-fiber diet and drink water.", "Avoid straining during bowel movements.", "Take warm sitz baths for 15-20 minutes.", "Use topical ointments if needed."],
    "Heart attack": ["Call emergency services immediately (911/15)!", "Chew an aspirin if recommended by dispatch.", "Rest quietly while waiting for responders.", "Stay calm and sit down."],
    "Varicose veins": ["Avoid standing or sitting for long periods.", "Elevate your legs above heart level when resting.", "Wear graduated compression stockings daily.", "Exercise regularly to improve circulation."],
    "Hypothyroidism": ["Take replacement medication on an empty stomach.", "Schedule regular blood tests for TSH.", "Consult your physician before new supplements.", "Maintain a balanced diet."],
    "Hyperthyroidism": ["Follow your prescribed antithyroid medication closely.", "Monitor your heart rate regularly.", "Ensure adequate intake of calcium.", "Limit excessive caffeine intake."],
    "Hypoglycemia": ["Consume 15-20 grams of fast-acting glucose.", "Recheck blood sugar levels after 15 minutes.", "Eat a small meal containing complex carbs.", "Notify your doctor if it occurs frequently."],
    "Osteoarthristis": ["Engage in regular, low-impact exercises.", "Maintain a healthy weight.", "Apply hot packs or cold packs.", "Use over-the-counter pain relievers."],
    "Arthritis": ["Consult a rheumatologist for a tailored plan.", "Incorporate anti-inflammatory foods.", "Balance active exercises with rest.", "Utilize joint protection techniques."],
    "(vertigo) Paroymsal Positional Vertigo": ["Avoid sudden head movements.", "Consult a specialist for the Epley maneuver.", "Sit down immediately when dizzy.", "Ensure your home is safe from tripping hazards."],
    "Acne": ["Wash your face gently with a mild cleanser.", "Avoid squeezing or picking at pimples.", "Use non-comedogenic skincare products.", "Apply targeted topical treatments."],
    "Urinary tract infection": ["Drink plenty of water to flush bacteria.", "Consult a professional for antibiotics.", "Avoid alcohol and caffeine.", "Urinate promptly when you feel the urge."],
    "Psoriasis": ["Keep your skin well-moisturized.", "Avoid dry or cold weather triggers.", "Follow your prescribed topical treatments.", "Identify and manage stressors."],
    "Impetigo": ["Consult a doctor for prescription ointment.", "Gently wash sores with soap and warm water.", "Cover sores with a clean bandage.", "Keep your nails short and clean."],
    "Gastroenteritis": ["Focus on replacing lost fluids.", "Eat small, bland meals (rice, bananas).", "Avoid dairy, caffeine, and fatty foods.", "Maintain strict handwashing habits."],
    "Bronchial Asthma": ["Keep your rescue inhaler with you at all times.", "Minimize exposure to asthma triggers.", "Follow your personalized action plan.", "Seek emergency care if breathing fails."],
    "Hypertension": ["Maintain a low-sodium, heart-healthy diet.", "Engage in moderate physical activity.", "Monitor blood pressure levels at home.", "Take medications consistently."],
    "Migraine": ["Rest in a quiet, dark, and cool room.", "Apply a cold compress to your forehead.", "Hydrate with small sips of water.", "Take abortive medications as directed."],
    "Cervical spondylosis": ["Practice good posture while working.", "Perform regular gentle neck stretching.", "Apply a heating pad or ice pack.", "Use a supportive neck pillow."],
    "Paralysis (brain hemorrhage)": ["Call emergency medical services immediately!", "Do not give anything to eat or drink.", "Help the person lie down on their side.", "Monitor breathing and responsiveness."],
    "Jaundice": ["Consult a physician promptly for liver tracking.", "Avoid all alcohol and unprescribed pills.", "Consume a nutritious, easily digestible diet.", "Monitor for abdominal swelling."],
    "Peptic ulcer diseae": ["Follow your prescribed proton pump inhibitors.", "Avoid NSAIDs like aspirin or ibuprofen.", "Limit spicy, fatty, or highly acidic items.", "Avoid smoking or alcohol consumption."],
    "AIDS": ["Consult an infectious disease specialist for ART.", "Adhere strictly to your daily medication schedule.", "Protect yourself from opportunistic infections.", "Schedule regular clinical follow-ups."],
    "Gerd": ["Eat smaller, more frequent meals.", "Avoid lying down for 3 hours after eating.", "Elevate the head of your bed.", "Limit fried dishes, chocolate, and caffeine."],
    "Chronic cholestasis": ["Consult a hepatologist for targeted diagnostic.", "Follow a low-fat diet.", "Use prescribed medications for itching.", "Monitor liver function tests regularly."],
    "Pure hypocholesterolemia": ["Consult your healthcare provider for causes.", "Adopt a balanced diet rich in healthy fats.", "Monitor your nutritional status regularly.", "Discuss any adjustments with your physician."],
    "Hepatitis A": ["Ensure plenty of bed rest.", "Stay hydrated by sipping water.", "Avoid all alcohol and over-the-counter pills.", "Wash hands thoroughly."],
    "Hepatitis B": ["Consult a specialist to check antiviral needs.", "Avoid alcohol and liver-toxic substances.", "Eat a balanced, healthy diet.", "Take precautions to prevent transmission."],
    "Hepatitis C": ["Consult a hepatologist to discuss DAA treatment.", "Avoid alcohol and check before new pills.", "Do not share personal items like razors.", "Monitor liver health indicators."],
    "Hepatitis D": ["Consult a liver specialist for monitoring.", "Adhere to treatment strategies for co-infection.", "Avoid alcohol and eat a nutrient-rich diet.", "Get regular blood tests."],
    "Hepatitis E": ["Get plenty of rest and stay hydrated.", "Avoid alcohol and unnecessary drugs.", "Seek immediate attention if pregnant.", "Ensure drinking water is clean."],
    "Alcoholic hepatitis": ["Stop consuming alcohol entirely.", "Consult a professional for support.", "Follow a high-protein, calorie-dense diet.", "Monitor closely for liver complications."],
    "Tuberculosis": ["Consult a specialist and follow your course.", "Complete the entire antibiotic regimen.", "Stay in a well-ventilated room initially.", "Maintain a balanced diet."]
}

@login_required
def index(request):
    lang = request.GET.get('lang', request.session.get('lang', 'fr'))
    request.session['lang'] = lang

    resultat = None
    description = None
    symptomes_coches = []
    top_predictions = []
    confiance_principale = 0
    recommendations = []
    error_message = None

    if request.method == 'POST':
        symptomes_coches = request.POST.getlist('symptomes')
        
        if not symptomes_coches:
            error_message = "Please select at least one clinical symptom." if lang == 'en' else "Veuillez sélectionner au moins un symptôme clinique."
        else:
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
            confiance_principale = int(predictions_associees[0][1] * 100)
            
            if lang == 'en':
                resultat = raw_resultat
                description = DESCRIPTIONS_MALADIES_EN.get(raw_resultat, "No description available.")
                recommendations = RECOMMENDATIONS_MALADIES_EN.get(raw_resultat, ["Consult a medical professional."])
            else:
                resultat = NOM_FRANCAIS_MALADIES.get(raw_resultat, raw_resultat)
                description = DESCRIPTIONS_MALADIES_FR.get(raw_resultat, "Aucune description disponible.")
                recommendations = RECOMMENDATIONS_MALADIES_FR.get(raw_resultat, ["Consultez un professionnel de la santé."])
            
            for mal, prob in predictions_associees[:3]:
                if prob > 0:
                    nom_affiche = mal if lang == 'en' else NOM_FRANCAIS_MALADIES.get(mal, mal)
                    top_predictions.append({'nom': nom_affiche, 'score': int(prob * 100)})

    liste_symptomes_traduits = []
    for symp in liste_symptomes:
        nom_propre_en = symp.replace('_', ' ').capitalize()
        nom_affiche = nom_propre_en if lang == 'en' else TRADUCTION_SYMPTOMES.get(symp, nom_propre_en)
        liste_symptomes_traduits.append({'cle_technique': symp, 'nom_affichage': nom_affiche})

    if lang == 'en':
        symptomes_coches_affichage = [s.replace('_', ' ').capitalize() for s in symptomes_coches]
        labels = {
            "title": "PathoPredict Analytics — Algorithmic Inference Platform",
            "btn_dataset": "Training Dataset", "btn_logout": "Sign Out",
            "kpi_patho": "Classified Pathologies", "kpi_vars": "Explanatory Variables",
            "kpi_obs": "Sampled Observations", "kpi_algo": "Ensemble Algorithm", "kpi_engine": "Inference Engine",
            "inf_title": "Algorithmic Inference Result", "inf_session": "User Session",
            "inf_class": "Estimated Primary Class :", "inf_conf": "Model Confidence Score",
            "inf_synth": "📌 Clinical Description Synthesis", "inf_risk": "Distribution Risk Spectrum (Top 3) :",
            "inf_rec": "🔬 Recommended Steps for Recovery :", "inf_warning": "⚠️ This automated prediction is based on statistical correlation algorithms. It is not a substitute for clinical medical evaluation.",
            "matrix_title": "Clinical Configuration Matrix", "matrix_desc": "Select and combine observed parameters to submit the feature vector to the predictive model.",
            "search_placeholder": "🔍 Filter clinical symptoms (e.g., pain, fever, cough)...", "active_params": "Active parameters for statistical analysis:",
            "grid_empty": "Model Parameters Index", "grid_match": "Search Results Matches", "btn_submit": "Execute Statistical Analysis",
            "off_title": "Source Dataset Matrix", "off_struct": "Variable Structure",
            "off_struct_desc": "Each column represents a unique clinical symptom binarized in the dataset: 1 indicates symptom presence, and 0 represents its absence.",
            "off_table_target": "Pathology Target", "off_table_itch": "itching", "off_table_rash": "skin_rash", "off_table_chill": "chills",
            "off_balance": "Dataset Uniform Balance", "off_balance_desc": "The classifier is trained on a balanced database of 4,920 records with 132 distinct symptoms. Classes have an exact uniform distribution, securing maximum mathematical stability for the decision tree estimators."
        }
    else:
        symptomes_coches_affichage = [TRADUCTION_SYMPTOMES.get(s, s.replace('_', ' ').capitalize()) for s in symptomes_coches]
        labels = {
            "title": "PathoPredict Analytics — Plateforme d'Inférence Algorithmique",
            "btn_dataset": "Base d'Entraînement", "btn_logout": "Se déconnecter",
            "kpi_patho": "Pathologies Classifiées", "kpi_vars": "Variables Explicatives",
            "kpi_obs": "Observations Échantillonnées", "kpi_algo": "Algorithme d'Ensemble", "kpi_engine": "Moteur d'Inférence",
            "inf_title": "Résultat de l'Inférence Algorithmique", "inf_session": "Session Utilisateur",
            "inf_class": "Classe Principale Estimée :", "inf_conf": "Indice de confiance du modèle",
            "inf_synth": "📌 Synthèse de la description clinique", "inf_risk": "Spectre de distribution des risques (Top 3) :",
            "inf_rec": "🔬 Conseils & actions de première urgence :", "inf_warning": "⚠️ Cette prédiction automatisée est basée sur des algorithmes de corrélation statistique. Elle ne se substitue en aucun cas à une évaluation médicale clinique.",
            "matrix_title": "Matrice de Configuration Clinique", "matrix_desc": "Sélectionnez et combinez les paramètres observés pour soumettre le vecteur de caractéristiques au modèle prédictif.",
            "search_placeholder": "🔍 Filtrer les symptômes cliniques (ex : douleur, fièvre, toux)...", "active_params": "Paramètres actifs pour l'analyse statistique :",
            "grid_empty": "Index des Paramètres du Modèle", "grid_match": "Résultats de la Recherche", "btn_submit": "Exécuter l'Analyse Statistique",
            "off_title": "Matrice de la Source de Données", "off_struct": "Structure des Variables",
            "off_struct_desc": "Chaque colonne représente un symptôme clinique unique binarisé dans le jeu de données : 1 indique la présence du symptôme, et 0 représente son absence.",
            "off_table_target": "Pathologie Cible", "off_table_itch": "Démangeaisons", "off_table_rash": "Éruption cutanée", "off_table_chill": "Frissons froids",
            "off_balance": "Équilibre Uniforme du Dataset", "off_balance_desc": "Le classifieur est entraîné sur une base de données équilibrée de 4 920 enregistrements avec 132 symptômes distincts. Les classes possèdent une distribution uniforme exacte, assurant une stabilité mathématique optimale pour les estimateurs de type arbre de décision."
        }

    context = {
        'liste_symptomes': liste_symptomes_traduits,
        'resultat': resultat,
        'description': description,
        'symptomes_coches': symptomes_coches_affichage,
        'symptomes_valeurs_brutes': symptomes_coches,
        'top_predictions': top_predictions,
        'confiance_principale': confiance_principale,
        'recommendations': recommendations,
        'error_message': error_message,
        'lang': lang,
        'labels': labels,
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
    lang = request.GET.get('lang', request.session.get('lang', 'fr'))
    request.session['lang'] = lang
    
    if request.user.is_authenticated:
        return redirect('index')
        
    if lang == 'en':
        labels = {
            "welcome": "Welcome to PathoPredict Analytics",
            "subtitle": "Advanced Diagnostic Platform Powered by Ensemble Learning Methods",
            "btn_enter": "Access Platform Dashboard",
            "footer": "National Institute of Statistics and Applied Economics — Biostatistics & Big Data Track"
        }
    else:
        labels = {
            "welcome": "Bienvenue sur PathoPredict Analytics",
            "subtitle": "Plateforme de Diagnostic Avancé Basée sur les Méthodes d'Apprentissage d'Ensemble",
            "btn_enter": "Accéder au Tableau de Bord",
            "footer": "Institut National de Statistique et d'Économie Appliquée — Filière Biostatistique & Big Data"
        }
        
    return render(request, 'prediction_app/intro.html', {'lang': lang, 'labels': labels})