import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Parte 1: Definir rutas ===
base_dir = os.getcwd()
test_dir = os.path.join(base_dir, 'test')
output_dir = test_dir  # Guardamos dentro de test/

# === Parte 2: Configuraciones de modelos ===
configurations = [
    {'ID': 0, 'patch_size': 16, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 1, 'patch_size': 16, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 2, 'patch_size': 16, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    {'ID': 3, 'patch_size': 8, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 4, 'patch_size': 8, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 5, 'patch_size': 8, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    {'ID': 6, 'patch_size': 4, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 7, 'patch_size': 4, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 8, 'patch_size': 4, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
]

# === Parte 3: Cargar resultados ===
all_results = []

for config in configurations:
    model_id = config['ID']
    model_name = f'P{config["patch_size"]}H{config["hidden_size"]}A{config["attention_heads"]}'
    result_csv = os.path.join(test_dir, model_name, f'{model_name}_metrics.csv')

    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
        df['Model_Name'] = model_name
        all_results.append(df)
        print(f"Cargado: {model_name}")
    else:
        print(f"No encontrado: {result_csv}")

if not all_results:
    print("No se encontraron archivos.")
    exit()

df_all = pd.concat(all_results, ignore_index=True)

# === Parte 4: Resumen de métricas promedio por modelo ===
df_summary = df_all.groupby('Model_Name')[['Accuracy', 'Mean_IoU', 'Mean_Dice', 'Inference_Time']].mean().reset_index()

def plot_bar(ax, values, labels, title, color):
    y_pos = list(range(len(values)))
    ax.barh(y_pos, values, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)

acc = df_summary.sort_values('Accuracy', ascending=False)
iou = df_summary.sort_values('Mean_IoU', ascending=False)
dice = df_summary.sort_values('Mean_Dice', ascending=False)
time = df_summary.sort_values('Inference_Time')

fig, axs = plt.subplots(1, 4, figsize=(24, 6))
plot_bar(axs[0], acc['Accuracy'], acc['Model_Name'], 'Accuracy', 'skyblue')
plot_bar(axs[1], iou['Mean_IoU'], iou['Model_Name'], 'Mean IoU', 'lightgreen')
plot_bar(axs[2], dice['Mean_Dice'], dice['Model_Name'], 'Mean Dice', 'coral')
plot_bar(axs[3], time['Inference_Time'], time['Model_Name'], 'Inference Time (s)', 'orange')
plt.tight_layout()

output_path = os.path.join(output_dir, 'comparativa_modelos.png')
plt.savefig(output_path)
plt.close()
print(f"Gráfica resumen guardada en: {output_path}")

# === Parte 5: Matrices de confusión simplificadas (barras) ===
def count_classes(class_str):
    if pd.isna(class_str):
        return set()
    class_str = str(class_str).strip()
    if class_str == '':
        return set()
    return set(class_str.split('|'))

confusion_summary = []

for model_name, group in df_all.groupby('Model_Name'):
    gt_classes_counts = []
    pred_classes_counts = []
    missing_classes_counts = []
    false_positive_counts = []

    for _, row in group.iterrows():
        gt_set = count_classes(row.get('GT_Classes', ''))
        pred_set = count_classes(row.get('Pred_Classes', ''))
        missing_set = count_classes(row.get('Missing_Classes', ''))
        fp_set = count_classes(row.get('False_Positive_Classes', ''))

        gt_classes_counts.append(len(gt_set))
        pred_classes_counts.append(len(pred_set))
        missing_classes_counts.append(len(missing_set))
        false_positive_counts.append(len(fp_set))

    confusion_summary.append({
        'Model_Name': model_name,
        'GT_Classes_Mean': np.mean(gt_classes_counts),
        'Pred_Classes_Mean': np.mean(pred_classes_counts),
        'Missing_Classes_Mean': np.mean(missing_classes_counts),
        'False_Positive_Classes_Mean': np.mean(false_positive_counts)
    })

df_confusion_summary = pd.DataFrame(confusion_summary)

fig2, ax2 = plt.subplots(figsize=(12, 8))
bar_width = 0.2
indices = np.arange(len(df_confusion_summary))

ax2.bar(indices, df_confusion_summary['GT_Classes_Mean'], width=bar_width, label='Clases GT promedio', color='blue')
ax2.bar(indices + bar_width, df_confusion_summary['Pred_Classes_Mean'], width=bar_width, label='Clases Predichas promedio', color='green')
ax2.bar(indices + 2*bar_width, df_confusion_summary['Missing_Classes_Mean'], width=bar_width, label='Clases Faltantes promedio', color='red')
ax2.bar(indices + 3*bar_width, df_confusion_summary['False_Positive_Classes_Mean'], width=bar_width, label='Falsos Positivos promedio', color='orange')

ax2.set_xticks(indices + 1.5*bar_width)
ax2.set_xticklabels(df_confusion_summary['Model_Name'], rotation=45, ha='right')
ax2.set_ylabel('Número promedio de clases')
ax2.set_title('Resumen de Matrices de Confusión por Modelo')
ax2.legend()

plt.tight_layout()
output_path2 = os.path.join(output_dir, 'matrices_confusion_resumen.png')
plt.savefig(output_path2)
plt.close()
print(f"Gráfica de matrices de confusión guardada en: {output_path2}")

# === Parte 6: Matrices de Confusión Visuales por Modelo (sin seaborn) ===
all_classes = list(map(str, range(20)))
num_classes = len(all_classes)
conf_matrix_dir = os.path.join(output_dir, 'matrices_confusion')
os.makedirs(conf_matrix_dir, exist_ok=True)

for model_name, group in df_all.groupby('Model_Name'):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for _, row in group.iterrows():
        gt = count_classes(row.get('GT_Classes', ''))
        pred = count_classes(row.get('Pred_Classes', ''))

        for gt_cls in gt:
            gt_idx = int(gt_cls)
            if gt_cls in pred:
                conf_matrix[gt_idx][gt_idx] += 1
        for pred_cls in pred:
            if pred_cls not in gt:
                pred_idx = int(pred_cls)
                conf_matrix[num_classes-1][pred_idx] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, cmap='Blues')

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(all_classes)
    ax.set_yticklabels(all_classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Clases predichas")
    ax.set_ylabel("Clases reales")
    ax.set_title(f"Matriz de Confusión - {model_name}")

    for i in range(num_classes):
        for j in range(num_classes):
            value = conf_matrix[i, j]
            if value > 0:
                ax.text(j, i, str(value), ha='center', va='center', color='black', fontsize=8)

    fig.tight_layout()
    matrix_path = os.path.join(conf_matrix_dir, f"conf_matrix_{model_name}.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"Matriz de confusión guardada para {model_name}: {matrix_path}")
