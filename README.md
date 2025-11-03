# Customer Segmentation & Churn Prediction for Telco

**Author:** Gilbert Urinzwenimana   


## 📋 Project Description

A machine learning solution that combines unsupervised and supervised learning to help telecommunications businesses understand customer segments and predict customer churn. The system uses K-means clustering to identify distinct customer groups based on behavior and characteristics, then applies classification models to predict which customers are at risk of leaving. This dual approach enables businesses to develop targeted retention strategies for high-risk segments, improving customer satisfaction and reducing revenue loss.

---

## 🎯 Problem Statement

**Business Challenge:** Many businesses face significant revenue losses due to customer churn caused by:
- Lack of understanding of customer segments
- Inability to identify at-risk customers early
- Generic retention strategies that don't address specific customer needs
- Poor customer satisfaction management

**Solution:** Develop ML algorithms to segment customers and predict churn risk, enabling personalized retention strategies.

---

## 🔧 Methodology

### Data Preprocessing & Exploration

**Data Quality Checks:**
- **Shape analysis:** Understanding dataset size (7043 rows × 33 columns)
- **Data types:** Verify numerical vs categorical features
- **Summary statistics:** Detect outliers via mean/max comparisons
- **Distribution analysis:** Check for imbalanced datasets
- **Missing values:** Handle nulls appropriately
- **Duplicates:** Remove redundant rows/columns
- **Encoding:** Label encoding for categorical variables
- **Scaling:** Standardization for numerical features
- **Feature selection:** Correlation analysis and visualization

### Model Development

**Part 1: Customer Segmentation (Unsupervised Learning)**

**Algorithm:** K-means Clustering

**Why K-means?**
- Suitable for dataset size (7043 samples, 33 features)
- Efficient with numerical features
- Scalable and interpretable

**Steps:**
1. **Scaling:** Apply StandardScaler() to normalize features (mean=0, std=1)
2. **Optimal K:** Use Elbow method to find best number of clusters
3. **Validation:** Silhouette score to evaluate cluster quality
4. **Model Building:** Fit K-means with optimal k

**Part 2: Churn Prediction (Supervised Learning)**

**Algorithms:**
1. **Random Forest Classifier** (Primary)
   - Combines multiple decision trees
   - Higher accuracy through ensemble
   - Captures non-linear relationships
   - Handles feature importance

2. **Logistic Regression** (Baseline)
   - Simple, interpretable model
   - Provides probability estimates
   - Baseline for comparison

**Evaluation:**
- Confusion matrix analysis
- Precision: Accuracy of positive predictions
- Recall: Coverage of actual positives
- F1-Score: Balance between precision and recall

### Integration Strategy

**Combined Approach:**
1. **Segment customers** using K-means based on characteristics
2. **Apply churn prediction** models to each segment
3. **Identify high-risk customers** within each segment
4. **Develop targeted strategies** for retention:
   - Personalized offers for high-value segments
   - Service improvements for dissatisfied segments
   - Proactive outreach for at-risk customers

**Expected Outcome:** Increased customer satisfaction and reduced churn rate

---

## 📊 Dataset

**Size:** 7,043 customers × 33 features

**Feature Categories:**
- **Demographics:** Age, gender, location
- **Account information:** Tenure, contract type, payment method
- **Service usage:** Internet, phone, streaming services
- **Billing:** Monthly charges, total charges
- **Target variable:** Churn (Yes/No)

---

## 🛠️ Technologies

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Preprocessing:** Scikit-learn (StandardScaler, LabelEncoder)
- **Clustering:** K-means (sklearn.cluster)
- **Classification:** Random Forest, Logistic Regression
- **Evaluation:** Confusion Matrix, Classification Report
- **Environment:** Python 3.x, Jupyter Notebook

---


---

## 🎓 Key Questions Answered

1. **Who are our customers?** Segment profiles and characteristics
2. **Why do customers leave?** Key churn drivers and patterns
3. **Which customers are at risk?** Churn probability scores
4. **What actions should we take?** Targeted retention strategies
5. **What's the ROI?** Expected reduction in churn and revenue impact

---

## 📊 Deliverables

1. **Jupyter Notebook** - Complete analysis with code
2. **Customer Segments** - K-means clustering results
3. **Churn Prediction Models** - Random Forest & Logistic Regression
4. **Evaluation Metrics** - Confusion matrix, precision, recall, F1-score
5. **Visualizations** - Cluster plots, feature importance, confusion matrix
6. **Business Recommendations** - Segment-specific retention strategies
7. **Final Report** - Insights and actionable recommendations

---

## 🔒 Ethical Considerations

**Privacy Protection:**
- Customer data anonymized to protect identity
- No personally identifiable information (PII) exposed
- Secure data handling throughout analysis
- Compliance with data protection regulations

**Fair Treatment:**
- Ensure retention strategies don't discriminate
- Transparent model decision-making
- Equal opportunity for all customer segments
- Avoid bias in segmentation and prediction

---

## 🚀 Business Applications

**Retention Strategy Development:**
- Design segment-specific offers and promotions
- Prioritize high-value customers at risk
- Allocate marketing budget efficiently

**Service Improvements:**
- Address pain points in dissatisfied segments
- Enhance features for high-engagement customers
- Optimize service bundles by segment

**Proactive Customer Management:**
- Early intervention for at-risk customers
- Personalized communication strategies
- Loyalty programs for engaged segments

**Revenue Optimization:**
- Reduce customer acquisition costs
- Increase customer lifetime value
- Improve profit margins through retention

---


## 👤 Author

**Gilbert Urinzwenimana**  
Carnegie Mellon University Africa

---

*This project demonstrates the power of combining unsupervised and supervised learning to solve real-world business problems, providing actionable insights for customer retention and revenue growth.*
