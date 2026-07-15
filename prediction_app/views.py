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
    "itching": "Démangeaisons",
    "skin_rash": "Éruption cutanée",
    "continuous_sneezing": "Éternuements continus",
    "shivering": "Frissons",
    "chills": "Courbatures / Frissons froids",
    "joint_pain": "Douleurs articulaires",
    "stomach_pain": "Maux d'estomac",
    "acidity": "Acidité gastrique",
    "ulcers_on_tongue": "Aphtes sur la langue",
    "muscle_wasting": "Fonte musculaire",
    "vomiting": "Vomissements",
    "burning_micturition": "Brûlure mictionnelle (en urinant)",
    "spotting_urination": "Traces de sang dans les urines",
    "fatigue": "Fatigue",
    "weight_gain": "Prise de poids",
    "anxiety": "Anxiété",
    "cold_hands_and_feets": "Mains et pieds froids",
    "mood_swings": "Changements d'humeur",
    "weight_loss": "Perte de poids",
    "restlessness": "Agitation",
    "lethargy": "Léthargie",
    "patches_in_throat": "Plaques dans la gorge",
    "irregular_sugar_level": "Taux de sucre irrégulier",
    "cough": "Toux",
    "high_fever": "Fièvre élevée",
    "sunken_eyes": "Yeux creux",
    "breathlessness": "Essoufflement",
    "sweating": "Transpiration excessive",
    "dehydration": "Déshydratation",
    "indigestion": "Indigestions",
    "headache": "Maux de tête",
    "yellowish_skin": "Peau jaunâtre",
    "dark_urine": "Urines foncées",
    "nausea": "Nausées",
    "loss_of_appetite": "Perte d'appétit",
    "pain_behind_the_eyes": "Douleur derrière les yeux",
    "back_pain": "Mal de dos",
    "constipation": "Constipation",
    "abdominal_pain": "Douleur abdominale",
    "diarrhoea": "Diarrhée",
    "mild_fever": "Fièvre légère",
    "yellow_urine": "Urines jaunes",
    "yellowing_of_eyes": "Jaunissement des yeux",
    "acute_liver_failure": "Insuffisance hépatique aiguë",
    "fluid_overload": "Surcharge de fluides",
    "swelling_of_stomach": "Gonflement de l'estomac",
    "swelled_lymph_nodes": "Ganglions lymphatiques gonflés",
    "malaise": "Malaise général",
    "blurred_and_distorted_vision": "Vision floue et déformée",
    "phlegm": "Crachats / Flegme",
    "throat_irritation": "Irritation de la gorge",
    "redness_of_eyes": "Rougeur des yeux",
    "sinus_pressure": "Pression sinusale",
    "runny_nose": "Écoulement nasal",
    "congestion": "Congestion nasale",
    "chest_pain": "Douleur thoracique",
    "weakness_in_limbs": "Faiblesse des membres",
    "fast_heart_rate": "Rythme cardiaque rapide",
    "pain_during_bowel_movements": "Douleur lors de la défécation",
    "pain_in_anal_region": "Douleur dans la région anale",
    "bloody_stool": "Selles sanglantes",
    "irritation_in_anus": "Irritation de l'anus",
    "neck_pain": "Douleur au cou",
    "dizziness": "Sensations de vertige",
    "cramps": "Crampes",
    "bruising": "Ecchymoses / Bleus",
    "obesity": "Obésité",
    "swollen_legs": "Jambes enflées",
    "swollen_blood_vessels": "Vaisseaux sanguins gonflés",
    "puffy_face_and_eyes": "Visage et yeux bouffis",
    "enlarged_thyroid": "Gorge / Thyroïde gonflée",
    "brittle_nails": "Ongles cassants",
    "swollen_extremeties": "Extrémités gonflées",
    "excessive_hunger": "Faim excessive",
    "extra_marital_contacts": "Rapports sexuels non protégés",
    "drying_and_tingling_lips": "Lèvres sèches et picotements",
    "slurred_speech": "Troubles de la parole",
    "knee_pain": "Douleur au genou",
    "hip_joint_pain": "Douleur à la hanche",
    "muscle_weakness": "Faiblesse musculaire",
    "stiff_neck": "Raideur de la nuque",
    "swelling_joints": "Gonflement des articulations",
    "movement_stiffness": "Rigidité des mouvements",
    "spinning_movements": "Sensations de rotation",
    "loss_of_balance": "Perte d'équilibre",
    "unsteadiness": "Instabilité physique",
    "weakness_of_one_body_side": "Faiblesse d'un côté du corps",
    "loss_of_smell": "Perte d'odorat",
    "bladder_discomfort": "Inconfort vésical",
    "foul_smell_of_urine": "Urine malodorante",
    "continuous_feel_of_urine": "Envie constante d'uriner",
    "passage_of_gases": "Passage de gaz",
    "internal_itching": "Démangeaisons internes",
    "toxic_look_(typhos)": "Faciès toxique",
    "depression": "Dépression",
    "irritability": "Irritabilité",
    "muscle_pain": "Douleurs musculaires",
    "altered_sensorium": "Altération de la conscience",
    "red_spots_over_body": "Taches rouges sur le corps",
    "belly_pain": "Maux de ventre",
    "abnormal_menstruation": "Règles anormales",
    "dischromic_patches": "Plaques décolorées",
    "watering_from_eyes": "Yeux larmoyants",
    "increased_appetite": "Augmentation de l'appétit",
    "polyuria": "Miction excessive (Polyurie)",
    "family_history": "Antécédents familiaux",
    "mucoid_sputum": "Crachats muqueux",
    "rusty_sputum": "Crachats rouillés",
    "lack_of_concentration": "Manque de concentration",
    "visual_disturbances": "Troubles visuels",
    "receiving_blood_transfusion": "Historique de transfusion sanguine",
    "receiving_unsterile_injections": "Injections non stériles",
    "coma": "Coma",
    "stomach_bleeding": "Saignement de l'estomac",
    "distention_of_abdomen": "Distension de l'abdomen",
    "history_of_alcohol_consumption": "Antécédents de consommation d'alcool",
    "blood_in_sputum": "Sang dans les crachats",
    "prominent_veins_on_calf": "Veines saillantes sur le mollet",
    "palpitations": "Palpitations",
    "painful_walking": "Marche douloureuse",
    "pus_filled_pimples": "Boutons de pus",
    "blackheads": "Points noirs",
    "scurring": "Cicatrices d'acné",
    "skin_peeling": "Desquamation de la peau",
    "silver_like_dusting": "Pellicules argentées (psoriasis)",
    "small_dents_in_nails": "Petits creux sur les ongles",
    "inflammatory_nails": "Inflammation des ongles",
    "blister": "Ampoules / Cloches",
    "red_sore_around_nose": "Plaies rouges autour du nez",
    "yellow_crust_ooze": "Suintement de croûtes jaunes"
}

DESCRIPTIONS_MALADIES = {
    "Drug Reaction": "Une réaction immunitaire indésirable de l'organisme à un médicament, provoquant généralement des éruptions cutanées, des démangeaisons ou de la fièvre.",
    "Malaria": "Une infection parasitaire transmise par les moustiques, entraînant de fortes fièvres, des frissons, des maux de tête et une transpiration intense.",
    "Allergy": "Une réaction exagérée du système immunitaire à une substance étrangère (poussière, pollen, aliments), provoquant éternuements, démangeaisons ou rougeurs.",
    "Diabetes": "Un trouble métabolique chronique caractérisé par un taux de sucre élevé dans le sang, résultant d'une production insuffisante d'insuline.",
    "Dengue": "Une infection virale transmise par les moustiques, provoquant une forte fièvre, d'intenses douleurs articulaires et musculaires et des éruptions cutanées.",
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

RECOMMENDATIONS_MALADIES = {
    "Drug Reaction": [
        "Arrêtez immédiatement de prendre le médicament suspecté.",
        "Consultez d'urgence un médecin ou rendez-vous dans un service d'évaluation médicale.",
        "Prenez des antihistaminiques si conseillé par un médecin pour apaiser les démangeaisons.",
        "Surveillez les signes graves comme des difficultés respiratoires ou un gonflement (appelez les urgences)."
    ],
    "Malaria": [
        "Consultez immédiatement un médecin pour réaliser un test de diagnostic rapide (frottis sanguin).",
        "Commencez sans délai le traitement antipaludique prescrit afin d'éviter toute complication grave.",
        "Reposez-vous et buvez beaucoup d'eau pour rester hydraté.",
        "Utilisez du paracétamol pour contrôler la fièvre selon les directives médicales."
    ],
    "Allergy": [
        "Identifiez et évitez l'exposition à l'allergène déclencheur.",
        "Utilisez des antihistaminiques en vente libre ou des sprays nasaux pour gérer les symptômes.",
        "Gardez un auto-injecteur d'épinéphrine (EpiPen) à portée de main si vous avez des antécédents d'anaphylaxie grave.",
        "Consultez un allergologue si les symptômes persistent ou s'aggravent."
    ],
    "Diabetes": [
        "Surveillez régulièrement votre glycémie.",
        "Respectez scrupuleusement votre traitement prescrit par votre médecin (insuline ou antidiabétiques oraux).",
        "Maintenez une alimentation structurée à faible indice glycémique et pratiquez une activité physique.",
        "Ayez toujours sur vous des glucides à action rapide (comme du jus ou des morceaux de sucre) pour traiter une hypoglycémie."
    ],
    "Dengue": [
        "Reposez-vous autant que possible et assurez-vous d'avoir une excellente hydratation.",
        "Prenez du paracétamol contre la fièvre et la douleur ; évitez l'aspirine ou l'ibuprofène qui augmentent les risques d'hémorragie.",
        "Protégez-vous des piqûres de moustiques pour éviter la propagation du virus.",
        "Surveillez attentivement l'apparition de signes d'alerte graves (vomissements persistants, saignements)."
    ],
    "Typhoid": [
        "Consultez immédiatement un médecin afin d'obtenir une prescription d'antibiothérapie adaptée.",
        "Ne buvez que de l'eau propre et sûre (bouteille scellée ou eau bouillie).",
        "Maintenez une hygiène rigoureuse des mains en les lavant fréquemment à l'eau et au savon.",
        "Consommez des repas chauds et entièrement cuits, évitez les aliments crus."
    ],
    "Fungal infection": [
        "Appliquez des crèmes ou pommades antifongiques locales prescrites ou recommandées.",
        "Gardez la zone de peau touchée propre, fraîche et parfaitement sèche.",
        "Évitez de partager vos effets personnels comme les serviettes, vêtements ou brosses à cheveux.",
        "Portez des vêtements amples et respirants en coton."
    ],
    "Common Cold": [
        "Reposez-vous au maximum pour permettre à votre organisme de récupérer.",
        "Hydratez-vous régulièrement en buvant de l'eau, des bouillons chauds ou des tisanes.",
        "Utilisez des sprays nasaux salins ou un humidificateur d'air pour soulager la congestion.",
        "Prenez du paracétamol ou de l'ibuprofène pour soulager les courbatures et la fièvre."
    ],
    "Pneumonia": [
        "Consultez d'urgence un médecin pour obtenir une prescription d'antibiotiques ou d'antiviraux adaptés.",
        "Restez strictement au lit et limitez tout effort physique.",
        "Buvez des liquides chauds pour aider à fluidifier et évacuer le mucus bronchique.",
        "Prenez des antipyrétiques si nécessaire et appelez les urgences en cas de détresse respiratoire."
    ],
    "Dimorphic hemmorhoids(piles)": [
        "Adoptez un régime riche en fibres (fruits, légumes, céréales complètes) et buvez beaucoup d'eau.",
        "Évitez de faire des efforts de poussée excessifs lors de la défécation.",
        "Prenez des bains de siège tièdes pendant 15 à 20 minutes, 2 à 3 fois par jour.",
        "Utilisez des crèmes locales apaisantes ou des analgésiques si l'inconfort persiste."
    ],
    "Heart attack": [
        "Appelez immédiatement les services d'urgence (Samu au 15)—il s'agit d'une urgence vitale.",
        "Croquez et avalez un comprimé d'aspirine si recommandé par le médecin régulateur des urgences.",
        "Restez immobile et assis en attendant les secours ; ne tentez pas de vous rendre à l'hôpital par vos propres moyens.",
        "Essayez de respirer calmement pour réduire la pression sur le muscle cardiaque."
    ],
    "Varicose veins": [
        "Évitez de rester debout ou assis de manière prolongée sans bouger.",
        "Surélevez vos jambes au-dessus du niveau du cœur dès que vous êtes au repos.",
        "Portez quotidiennement des bas de contention adaptés.",
        "Pratiquez une activité physique régulière (marche, vélo) pour stimuler la circulation veineuse."
    ],
    "Hypothyroidism": [
        "Prenez votre traitement hormonal quotidien (lévothyroxine) strictement à jeun le matin.",
        "Effectuez régulièrement vos bilans sanguins pour contrôler vos taux d'hormone TSH.",
        "Consultez votre médecin avant de prendre de nouveaux compléments alimentaires qui pourraient altérer l'absorption.",
        "Adoptez un régime équilibré pour aider à stabiliser votre poids et votre niveau d'énergie."
    ],
    "Hyperthyroidism": [
        "Prenez rigoureusement vos antithyroïdiens de synthèse ou bêtabloquants prescrits.",
        "Surveillez régulièrement votre rythme cardiaque et signalez toute palpitation à votre médecin.",
        "Veillez à avoir des apports suffisants en calcium et vitamine D pour protéger votre densité osseuse.",
        "Limitez le café et autres excitants afin de réduire la nervosité et la tachycardie."
    ],
    "Hypoglycemia": [
        "Consommez immédiatement 15 à 20 g de sucre rapide (jus de fruit, morceaux de sucre, bonbons).",
        "Contrôlez à nouveau votre glycémie après 15 minutes et répétez la prise de sucre si elle reste basse.",
        "Prenez une collation ou un repas contenant des glucides complexes une fois le taux normalisé.",
        "Consultez votre médecin pour ajuster vos doses d'insuline ou de antidiabétiques oraux."
    ],
    "Osteoarthristis": [
        "Pratiquez régulièrement des exercices à faible impact (natation, cyclisme) pour renforcer les muscles de soutien.",
        "Maintenez un poids de forme pour limiter la pression exercée sur les articulations porteuses.",
        "Appliquez du chaud contre la raideur matinale ou du froid en cas de poussée inflammatoire douloureuse.",
        "Utilisez des antalgiques ou des gels anti-inflammatoires locaux selon les recommandations de votre médecin."
    ],
    "Arthritis": [
        "Consultez un rhumatologue pour mettre en place un plan de traitement de fond personnalisé.",
        "Incorporez des aliments aux vertus anti-inflammatoires dans vos repas quotidiens.",
        "Alternez des périodes d'exercices modérés et de repos complet lors des poussées douloureuses.",
        "Utilisez des aides ergonomiques et ménagez vos articulations lors des tâches ménagères."
    ],
    "(vertigo) Paroymsal Positional Vertigo": [
        "Évitez les mouvements brusques de la tête ainsi que l'extension forcée du cou vers l'arrière.",
        "Consultez un kinésithérapeute spécialisé pour réaliser une manœuvre libératoire (manœuvre d'Epley).",
        "Asseyez-vous immédiatement dès les premiers signes de rotation pour éviter une chute.",
        "Sécurisez votre domicile en retirant les tapis ou obstacles au sol."
    ],
    "Acne": [
        "Nettoyez votre visage en douceur deux fois par jour avec un gel nettoyant doux non comédogène.",
        "Évitez absolument de percer ou presser vos boutons pour ne pas propager l'infection ou créer des cicatrices.",
        "Utilisez des cosmétiques et crèmes hydratantes mentionnant la formule 'non comédogène'.",
        "Appliquez des soins locaux ciblés contenant de l'acide salicylique ou du peroxyde de benzoyle."
    ],
    "Urinary tract infection": [
        "Buvez d'importantes quantités d'eau pour aider à éliminer mécaniquement les bactéries par les urines.",
        "Consultez un médecin rapidement pour obtenir une prescription d'antibiotiques adaptés.",
        "Évitez les boissons irritantes comme l'alcool, le café ou les boissons très sucrées.",
        "Urinez dès que l'envie se fait sentir et essuyez-vous toujours d'avant en arrière."
    ],
    "Psoriasis": [
        "Hydratez très généreusement votre peau au quotidien avec des crèmes riches ou des émollients.",
        "Évitez autant que possible les agressions cutanées et l'exposition prolongée au froid sec.",
        "Suivez scrupuleusement votre traitement prescrit (dermocorticoïdes, photothérapie).",
        "Identifiez et gérez vos facteurs de stress, qui sont de puissants déclencheurs de poussées de psoriasis."
    ],
    "Impetigo": [
        "Consultez un médecin afin d'obtenir un traitement antibiotique local en pommade ou par voie orale.",
        "Nettoyez délicatement les zones lésées à l'eau tiède et au savon pour éliminer les croûtes.",
        "Couvrez les lésions d'une compresse stérile pour éviter l'auto-inoculation ou la transmission.",
        "Gardez des ongles courts et propres pour limiter le grattage."
    ],
    "Gastroenteritis": [
        "Privilégiez la réhydratation par de petites gorgées fréquentes de solutés de réhydratation ou d'eau salée.",
        "Dès l'arrêt des vomissements, reprenez une alimentation légère et fade (riz, banane, compote).",
        "Évitez temporairement les produits laitiers, les graisses, le café et les épices.",
        "Lavez-vous minutieusement les mains au savon pour protéger votre entourage de la contagion."
    ],
    "Bronchial Asthma": [
        "Gardez en permanence sur vous votre inhalateur de secours à action rapide (bronchodilatateur).",
        "Identifiez et limitez les facteurs déclencheurs d'essoufflement (poussière, fumée, air froid, pollens).",
        "Prenez régulièrement votre traitement de fond quotidien même en l'absence de crise.",
        "Appelez immédiatement les urgences si votre respiration ne s'améliore pas après utilisation de l'inhalateur."
    ],
    "Hypertension": [
        "Adoptez une alimentation pauvre en sel et riche en fruits et légumes.",
        "Pratiquez une activité d'endurance modérée (comme la marche rapide) au moins 30 minutes par jour.",
        "Mesurez régulièrement votre tension artérielle au calme à l'aide d'un tensiomètre automatique.",
        "Prenez vos traitements antihypertenseurs tous les jours sans interruption."
    ],
    "Migraine": [
        "Dès les premiers signes, allongez-vous dans une pièce sombre, fraîche et parfaitement silencieuse.",
        "Appliquez une compresse froide ou une poche de glace sur votre front ou vos tempes.",
        "Buvez de petites gorgées d'eau et évitez les repas lourds ou les aliments déclencheurs.",
        "Prenez votre traitement spécifique de crise (triptans ou anti-inflammatoires) dès le début du mal de tête."
    ],
    "Cervical spondylosis": [
        "Veillez à maintenir une bonne posture assise, notamment lors du travail sur ordinateur.",
        "Réalisez de légers étirements du cou et des exercices de renforcement musculaire de la nuque.",
        "Appliquez une source de chaleur douce sur la zone douloureuse pour détendre les tensions musculaires.",
        "Utilisez un oreiller ergonomique adapté pour soutenir l'alignement de vos cervicales durant la nuit."
    ],
    "Paralysis (brain hemorrhage)": [
        "Appelez immédiatement les urgences médicales (Samu au 15)—chaque minute compte.",
        "Ne donnez absolument rien à manger ou à boire à la personne.",
        "Allongez la personne sur le côté (position latérale de sécurité) si elle est inconsciente mais respire.",
        "Surveillez en continu sa respiration en attendant l'arrivée rapide des secours médicaux."
    ],
    "Jaundice": [
        "Consultez sans tarder un médecin pour diagnostiquer et traiter la cause sous-jacente liée au foie.",
        "Proscrivez totalement l'alcool ainsi que l'automédication pour ne pas fatiguer davantage le foie.",
        "Reposez-vous et assurez-vous d'avoir une alimentation saine, équilibrée et facile à digérer.",
        "Signalez immédiatement au médecin l'apparition de troubles de la conscience ou de gonflements abdominaux."
    ],
    "Peptic ulcer diseae": [
        "Prenez consciencieusement votre traitement anti-acide prescrit (inhibiteurs de la pompe à protons).",
        "Évitez la prise d'anti-inflammatoires non stéroïdiens (comme l'ibuprofène ou l'aspirine) sans avis médical.",
        "Limitez les aliments irritants comme les plats très épicés, acides ou trop gras.",
        "Supprimez le tabac et l'alcool, qui ralentissent grandement la cicatrisation de l'estomac."
    ],
    "AIDS": [
        "Consultez régulièrement votre médecin spécialiste pour suivre et ajuster votre traitement antirétroviral (TRV).",
        "Prenez vos médicaments chaque jour à heure fixe afin d'assurer l'efficacité du contrôle viral.",
        "Protégez-vous des infections opportunistes par une hygiène rigoureuse et des vaccins à jour.",
        "Effectuez régulièrement vos bilans sanguins pour suivre vos taux de lymphocytes CD4 et de charge virale."
    ],
    "Gerd": [
        "Prenez des repas plus légers et fractionnés plutôt que des repas copieux et gras.",
        "Attendez au moins 2 à 3 heures après votre repas avant de vous allonger ou d'aller dormir.",
        "Surélevez la tête de votre lit de quelques centimètres.",
        "Évitez les aliments favorisant le reflux (chocolat, menthe, café, boissons gazeuses, graisses)."
    ],
    "Chronic cholestasis": [
        "Consultez un hépatologue pour le suivi régulier de l'écoulement biliaire et des fonctions du foie.",
        "Adoptez un régime pauvre en graisses et discutez d'une supplémentation en vitamines liposolubles (A, D, E, K).",
        "Utilisez les traitements prescrits par votre médecin pour atténuer les fortes démangeaisons cutanées.",
        "Effectuez régulièrement vos prises de sang pour surveiller le bilan hépatique."
    ],
    "Pure hypocholesterolemia": [
        "Consultez votre médecin traitant afin d'explorer la cause sous-jacente (hyperthyroïdie, malabsorption).",
        "Adoptez un régime alimentaire varié incluant de bonnes graisses d'origine végétale (huile d'olive, avocat).",
        "Faites un suivi nutritionnel régulier pour éviter les carences en vitamines et minéraux.",
        "Discutez de vos résultats cliniques avec votre praticien pour adapter votre prise en charge globale."
    ],
    "Hepatitis A": [
        "Restez au repos complet et évitez toute forme d'effort physique intense.",
        "Hydratez-vous en buvant fréquemment de l'eau ou des solutions de réhydratation.",
        "Proscrivez l'alcool et demandez l'avis d'un médecin avant de prendre du paracétamol.",
        "Lavez-vous soigneusement les mains après chaque passage aux toilettes pour éviter de contaminer vos proches."
    ],
    "Hepatitis B": [
        "Consultez un hépatologue pour déterminer la nécessité d'un traitement antiviral.",
        "Évitez impérativement les boissons alcoolisées afin de préserver votre foie d'un stress supplémentaire.",
        "Consommez des repas équilibrés et observez un rythme de repos suffisant.",
        "Prenez des précautions lors des rapports sexuels pour éviter de transmettre le virus à vos partenaires."
    ],
    "Hepatitis C": [
        "Consultez un spécialiste pour bénéficier des nouveaux antiviraux à action directe (guérison en quelques semaines).",
        "Évitez la consommation d'alcool et ne prenez aucun traitement sans avis de votre médecin.",
        "Ne partagez aucun objet d'hygiène personnelle pouvant contenir des traces de sang (rasoir, coupe-ongles).",
        "Faites surveiller régulièrement l'état de votre foie par des examens biologiques et échographiques."
    ],
    "Hepatitis D": [
        "Consultez un spécialiste pour un suivi conjoint des infections par l'hépatite B et l'hépatite D.",
        "Respectez scrupuleusement le protocole thérapeutique ciblant la co-infection virale.",
        "Adoptez un mode de vie sain, sans alcool, pour soutenir la régénération du foie.",
        "Faites des contrôles réguliers de charge virale et d'enzymes hépatiques."
    ],
    "Hepatitis E": [
        "Assurez-vous un repos physique de qualité et buvez beaucoup d'eau pour éliminer les toxines.",
        "Évitez de consommer de l'alcool et limitez l'usage de médicaments non prescrits par votre médecin.",
        "Consultez immédiatement un médecin si vous êtes enceinte, l'hépatite E pouvant s'avérer très grave dans ce cas.",
        "Consommez exclusivement de l'eau potable sûre et des aliments parfaitement cuits."
    ],
    "Alcoholic hepatitis": [
        "Arrêtez de manière totale, définitive et immédiate toute consommation de boissons alcoolisées.",
        "Faites-vous accompagner par un médecin ou un addictologue pour faciliter le sevrage en toute sécurité.",
        "Suivez un programme nutritionnel riche en protéines et en calories pour combler les carences.",
        "Surveillez attentivement l'apparition de complications de l'insuffisance hépatique."
    ],
    "Tuberculosis": [
        "Prenez rigoureusement chaque jour l'ensemble des antibiotiques prescrits par votre médecin spécialiste.",
        "Ne stoppez jamais votre traitement de manière anticipée afin de prévenir l'apparition de bactéries résistantes.",
        "Restez isolé dans une pièce bien aérée et portez un masque chirurgical au début du traitement pour protéger autrui.",
        "Adoptez une alimentation fortifiante et riche en calories pour soutenir vos défenses immunitaires."
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
            error_message = "Veuillez sélectionner au moins un symptôme clinique avant d'exécuter l'analyse statistique."
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
            description = DESCRIPTIONS_MALADIES.get(raw_resultat, "Aucune description disponible.")
            confiance_principale = int(predictions_associees[0][1] * 100)
            
            # Pull clinical recommendations based on the predicted target
            recommendations = RECOMMENDATIONS_MALADIES.get(raw_resultat, ["Consultez un professionnel de la santé pour obtenir des conseils personnalisés."])
            
            for mal, prob in predictions_associees[:3]:
                if prob > 0:
                    top_predictions.append({
                        'nom': NOM_FRANCAIS_MALADIES.get(mal, mal),
                        'score': int(prob * 100)
                    })

    # -------------------------------------------------------------
    # TRANSFORMATION FRANÇAIS : Dictionnaires pour l'affichage HTML
    # -------------------------------------------------------------
    liste_symptomes_traduits = []
    for symp in liste_symptomes:
        nom_propre_en = symp.replace('_', ' ').capitalize()
        nom_fr = TRADUCTION_SYMPTOMES.get(symp, nom_propre_en)
        liste_symptomes_traduits.append({
            'cle_technique': symp,   # Valeur technique en anglais (ex: "itching")
            'nom_affichage': nom_fr  # Libellé en français (ex: "Démangeaisons")
        })

    # Traduction en français des symptômes cochés pour la restitution sur l'interface
    symptomes_coches_fr = [TRADUCTION_SYMPTOMES.get(s, s.replace('_', ' ').capitalize()) for s in symptomes_coches]

    context = {
        'liste_symptomes': liste_symptomes_traduits,
        'resultat': resultat,
        'description': description,
        'symptomes_coches': symptomes_coches_fr,
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