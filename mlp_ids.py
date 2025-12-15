import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data (comma-separated)
train_data = pd.read_csv('data/KDDTrain%2B.txt', header=None, sep=',')
test_data = pd.read_csv('data/KDDTest%2B.txt', header=None, sep=',')

# Columns (41 features + attack + difficulty = 43 total)
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
           'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
           'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
           'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate', 'attack', 'difficulty']
train_data.columns = columns
test_data.columns = columns

# Step 2: Preprocess (encode categoricals, scale)
categorical_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Binary classification: normal (0) vs attack (1)
train_data['attack'] = train_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['attack'] = test_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)

X_train = train_data.drop(['attack', 'difficulty'], axis=1)
y_train = train_data['attack']
X_test = test_data.drop(['attack', 'difficulty'], axis=1)
y_test = test_data['attack']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Confusion Matrix Visual
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()