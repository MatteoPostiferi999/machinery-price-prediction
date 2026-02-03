import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def deep_forensic_analysis(df_raw, worst_ids, good_ids=None):
    """
    Confronta i record 'Worst' con il resto del dataset per trovare pattern nascosti.
    """
    print("\n🕵️‍♂️ DEEP FORENSIC ANALYSIS: Caccia alle Anomalie")
    print("="*80)
    
    # 1. Etichettiamo i dati
    # Creiamo una colonna 'Status': 'Critical Error' vs 'Normal'
    df_analysis = df_raw.copy()
    df_analysis['Status'] = 'Normal'
    df_analysis.loc[df_analysis['Sales ID'].isin(worst_ids), 'Status'] = 'Critical Error'
    
    worst_df = df_analysis[df_analysis['Status'] == 'Critical Error']
    normal_df = df_analysis[df_analysis['Status'] == 'Normal']
    
    print(f"Analisi su {len(worst_df)} Errori Critici vs {len(normal_df)} Normali.")
    
    # 2. Analisi Colonne Categoriche (Sospetti Principali)
    # Cerchiamo colonne dove la distribuzione è MOLTO diversa
    categorical_cols = ['UsageBand', 'ProductGroup', 'Drive_System', 'Enclosure', 
                        'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 
                        'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 
                        'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 
                        'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 
                        'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 
                        'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 
                        'Travel_Controls', 'Differential_Type', 'Steering_Controls', 'state', 'datasource']
    
    suspicious_features = []
    
    print("\n🔍 CONFRONTO DISTRIBUZIONI (Top Anomalie):")
    print(f"{'Feature':<25} | {'Valore':<20} | {'% nei PEGGIORI':<15} | {'% nei NORMALI':<15} | {'Multiplier':<10}")
    print("-" * 95)
    
    for col in categorical_cols:
        if col not in df_analysis.columns: continue
        
        # Calcola frequenze
        worst_counts = worst_df[col].value_counts(normalize=True)
        normal_counts = normal_df[col].value_counts(normalize=True)
        
        # Confronta ogni valore della categoria
        for val in worst_counts.index:
            freq_bad = worst_counts[val]
            freq_good = normal_counts.get(val, 0)
            
            # Filtri per evitare rumore:
            # 1. Deve essere rilevante nei 'bad' (>5%)
            # 2. Deve essere molto più frequente che nei 'good' (>2x)
            if freq_bad > 0.05 and freq_good > 0.001:
                ratio = freq_bad / freq_good
                if ratio > 2.0:
                    print(f"{col:<25} | {str(val)[:20]:<20} | {freq_bad:>14.1%} | {freq_good:>14.1%} | {ratio:>9.1f}x")
                    suspicious_features.append((col, val, ratio))

    # 3. Analisi Valori Mancanti (Nulls)
    # A volte l'errore è causato da dati mancanti specifici
    print("\n🔍 ANALISI VALORI MANCANTI (Nulls):")
    print(f"{'Feature':<25} | {'% Null nei PEGGIORI':<20} | {'% Null nei NORMALI':<20} | {'Diff'}")
    print("-" * 80)
    
    for col in df_analysis.columns:
        null_bad = worst_df[col].isnull().mean()
        null_good = normal_df[col].isnull().mean()
        
        if abs(null_bad - null_good) > 0.2: # Se la differenza è > 20%
            print(f"{col:<25} | {null_bad:>19.1%} | {null_good:>19.1%} | {null_bad-null_good:>+5.1%}")

    return suspicious_features

# =============================================================================
# ESEMPIO DI ESECUZIONE (Copia questo pezzo nel tuo main.py o notebook)
# =============================================================================
# 1. Carica la lista dei SalesID dei peggiori errori (che hai appena incollato)
#    (Se hai salvato il CSV "reports/worst_errors_forensic.csv", caricalo!)
# 
# df_worst = pd.read_csv("reports/worst_errors_forensic.csv")
# worst_ids = df_worst['SalesID'].tolist()
#
# 2. Chiama la funzione passando il dataset grezzo ORIGINALE
# deep_forensic_analysis(df, worst_ids)