# Reducing_Customer_Turnover_by_Analyzing_Financial_Behaviors

<p align="center">
<img src="https://user-images.githubusercontent.com/61653147/230792333-6e699924-2e6a-49a8-92c8-a92e5b3b4dec.png" width="600" height="600" />
</p>

Companies across all industries often rely on subscription products as their main source of revenue. These products can take various forms, from a one-size-fits-all subscription to multi-level memberships. However, regardless of the subscription model or industry, companies aim to minimize customer churn (i.e., subscription cancellations). To achieve this, companies need to identify behavioral patterns that contribute to disengagement with the product and take proactive steps to re-engage these customers.

* Market: The target audience is the company's entire subscription base, and retaining these customers is the main goal.

* Product: The company's existing subscription products can provide value that customers may have forgotten or not yet realized.

* Goal: The objective of this model is to predict which users are likely to churn so that the company can focus on re-engaging them with the product. This can be achieved through email reminders about the product's benefits, focusing on new features or features that the user has shown to value.

## Business Problem 

* The company is a fintech provider of a subscription-based product that helps users manage their bank accounts, receive personalized coupons, and stay up-to-date with the latest low-APR loans. We also offer resources to help users save money, such as videos on tax-saving techniques and free courses on financial health.

* Our goal is to identify users who are likely to cancel their subscription so that we can create new features that align with their interests and increase their engagement with our product. By doing so, we hope to prevent churn and retain our user base.

## Data-Set

| Column Name          | Explanation                                            | Data Type |
|----------------------|--------------------------------------------------------|-----------|
| entry_id             | Unique identifier for loan application                 | int64     |
| age                  | Age of the loan applicant                              | int64     |
| pay_schedule         | Frequency of pay schedule (e.g. weekly, biweekly, etc.) | object    |
| home_owner           | Flag indicating if the applicant owns a home           | int64     |
| income               | Annual income of the loan applicant                     | int64     |
| months_employed      | Number of months employed at current job                | int64     |
| years_employed       | Number of years employed in total                       | int64     |
| current_address_year | Number of years at current address                      | int64     |
| personal_account_m   | Number of months since applicant opened personal account| int64     |
| personal_account_y   | Number of years since applicant opened personal account | int64     |
| has_debt             | Flag indicating if the applicant has any debt          | int64     |
| amount_requested     | Requested loan amount                                   | int64     |
| risk_score           | Risk score assigned by the lender                       | int64     |
| risk_score_2         | Risk score assigned by other credit model               | float64   |
| risk_score_3         | Risk score assigned by other credit model               | float64   |
| risk_score_4         | Risk score assigned by other credit model               | float64   |
| risk_score_5         | Risk score assigned by other credit model               | float64   |
| ext_quality_score    | External quality score                                  | float64   |
| ext_quality_score_2  | External quality score assigned by other model          | float64   |
| inquiries_last_month | Number of credit inquiries in the last month            | int64     |
| e_signed             | Flag indicating if the loan was e-signed                | int64     |



## Lessons Learned

* An accuracy of 0.65 is relatively low, which suggests that the model may not be performing well. 
* The Recall and Precision values are moderate, indicating that the model is not performing poorly or particularly well in terms of correctly identifying positive cases and minimizing false positives. 
* The F1 score combines Recall and Precision into a single metric, and the value of 0.68 indicates that the model is performing moderately well in this respect. Finally, the AUC of 0.71 is also moderate, indicating that the model's predictions are somewhat better than random. 

## 🚀 About Me


🔭 I’m currently working on Data Science

🌱 I’m currently learning Machine Learning

📫 How to reach me anilcogalan@outlook.com

💬 Ask me about Data Science

⚡I adore camping

🌐 Socials:

🔗 LinkedIn : https://www.linkedin.com/in/anilcogalan/ 

🔗 Medium : https://medium.com/@anilcogalan
