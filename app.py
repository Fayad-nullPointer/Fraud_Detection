import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define any custom functions that might be in the pipeline
def log_transform(x):
    """Log transformation function that might be used in the pipeline"""
    return np.log1p(x)

# Add custom functions to main module for pickle compatibility
import __main__
__main__.log_transform = log_transform

# Load the trained pipeline
@st.cache_resource
def load_pipeline():
    try:
        # Try different file names and loading methods
        file_paths = [
            'fraud_detection_pipeline.pkl',
            # 'fraud_detection_model.pkl'
        ]
        
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as file:
                    pipeline = joblib.load(file)
                st.success(f"Pipeline loaded successfully from {file_path}!")
                return pipeline
            except FileNotFoundError:
                continue
            except Exception as e1:
                try:
                    with open(file_path, 'rb') as file:
                        pipeline = pickle.load(file)
                    st.success(f"Pipeline loaded successfully from {file_path} using pickle!")
                    return pipeline
                except Exception as e2:
                    st.warning(f"Failed to load {file_path}: {str(e1)[:100]}")
                    continue
        
        st.error("Could not find or load the pipeline file. Please ensure 'fraud_detection_pipeline.pkl' exists.")
        return None
        
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Detection System")
    st.markdown("### Enter transaction details to detect potential fraud")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    if pipeline is None:
        st.error("Cannot proceed without a trained pipeline.")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.title("📊 About")
        st.markdown("""
        **Fraud Detection System**
        
        This app uses a machine learning pipeline to detect fraudulent transactions.
        
        **Transaction Types:**
        - **CASH_IN**: Money deposited into account
        - **CASH_OUT**: Money withdrawn from account  
        - **DEBIT**: Direct deduction for purchases
        - **PAYMENT**: Payment for goods/services
        - **TRANSFER**: Money transfer between accounts
        
        **How it works:**
        1. Enter transaction details
        2. Pipeline processes and analyzes data
        3. Get fraud prediction with confidence
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit 🎈")
    
    # Test samples section
    with st.expander("🧪 Test with Suspicious Transaction Samples", expanded=False):
        st.markdown("**Click any button below to load a potentially fraudulent transaction pattern:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚨 Large Cash Out"):
                st.session_state.test_sample = {
                    'step': 180,
                    'type': 'CASH_OUT',
                    'amount': 9999999.99,
                    'oldbalance_org': 10000000.00,
                    'newbalance_orig': 0.01,
                    'oldbalance_dest': 0.0,
                    'newbalance_dest': 9999999.99
                }
        
        with col2:
            if st.button("💸 Suspicious Transfer"):
                st.session_state.test_sample = {
                    'step': 350,
                    'type': 'TRANSFER',
                    'amount': 5000000.00,
                    'oldbalance_org': 5000000.00,
                    'newbalance_orig': 0.00,
                    'oldbalance_dest': 1000.00,
                    'newbalance_dest': 5001000.00
                }
        
        with col3:
            if st.button("🔄 Round Amount Transfer"):
                st.session_state.test_sample = {
                    'step': 120,
                    'type': 'TRANSFER',
                    'amount': 1000000.00,
                    'oldbalance_org': 1500000.00,
                    'newbalance_orig': 500000.00,
                    'oldbalance_dest': 0.0,
                    'newbalance_dest': 0.0
                }
        
        # Second row of test samples
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⚡ Zero Balance Fraud"):
                st.session_state.test_sample = {
                    'step': 600,
                    'type': 'CASH_OUT',
                    'amount': 354208.00,
                    'oldbalance_org': 354208.00,
                    'newbalance_orig': 0.00,
                    'oldbalance_dest': 0.0,
                    'newbalance_dest': 0.0
                }
        
        with col2:
            if st.button("💰 High Amount Payment"):
                st.session_state.test_sample = {
                    'step': 400,
                    'type': 'PAYMENT',
                    'amount': 8888888.88,
                    'oldbalance_org': 9000000.00,
                    'newbalance_orig': 111111.12,
                    'oldbalance_dest': 0.0,
                    'newbalance_dest': 0.0
                }
        
        with col3:
            if st.button("🔄 Reset to Normal"):
                st.session_state.test_sample = {
                    'step': 43,
                    'type': 'CASH_OUT',
                    'amount': 107282.04,
                    'oldbalance_org': 10773.00,
                    'newbalance_orig': 0.00,
                    'oldbalance_dest': 480840.82,
                    'newbalance_dest': 588122.86
                }
    
    # Main input form
    st.markdown("### Transaction Information")
    
    with st.form("fraud_detection_form"):
        # Get test sample values if available
        test_sample = st.session_state.get('test_sample', {})
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            
            step = st.number_input(
                "Step (Time Unit)", 
                min_value=1, 
                max_value=744, 
                value=test_sample.get('step', 43),
                help="1 step = 1 hour, max 744 steps (30 days simulation)"
            )
            
            # Get the index for the transaction type
            transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            default_type = test_sample.get('type', 'CASH_OUT')
            type_index = transaction_types.index(default_type) if default_type in transaction_types else 1
            
            transaction_type = st.selectbox(
                "Transaction Type", 
                transaction_types,
                index=type_index,
                help="Select the type of transaction"
            )
            
            amount = st.number_input(
                "Transaction Amount", 
                min_value=0.01, 
                value=test_sample.get('amount', 107282.04),
                format="%.2f",
                help="Amount of the transaction in local currency"
            )
        
        with col2:
            st.subheader("Account Balances")
            
            oldbalance_org = st.number_input(
                "Origin Account - Old Balance", 
                min_value=0.0, 
                value=test_sample.get('oldbalance_org', 10773.00),
                format="%.2f",
                help="Initial balance before transaction"
            )
            
            newbalance_orig = st.number_input(
                "Origin Account - New Balance", 
                min_value=0.0, 
                value=test_sample.get('newbalance_orig', 0.00),
                format="%.2f",
                help="New balance after transaction"
            )
            
            oldbalance_dest = st.number_input(
                "Destination Account - Old Balance", 
                min_value=0.0, 
                value=test_sample.get('oldbalance_dest', 480840.82),
                format="%.2f",
                help="Initial balance of recipient (no info for Merchants starting with M)"
            )
            
            newbalance_dest = st.number_input(
                "Destination Account - New Balance", 
                min_value=0.0, 
                value=test_sample.get('newbalance_dest', 588122.86),
                format="%.2f",
                help="New balance of recipient after transaction"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔍 Analyze Transaction", 
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Prepare data for prediction
            prediction_data = prepare_prediction_data(
                step, transaction_type, amount, 
                oldbalance_org, newbalance_orig, 
                oldbalance_dest, newbalance_dest
            )
            
            # Make prediction
            try:
                with st.spinner('Analyzing transaction...'):
                    prediction = pipeline.predict(prediction_data)[0]
                    
                    # Try to get prediction probabilities
                    try:
                        prediction_proba = pipeline.predict_proba(prediction_data)[0]
                        has_proba = True
                    except:
                        has_proba = False
                
                # Display results
                display_results(prediction, prediction_proba if has_proba else None)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check if your input data format matches the trained model.")

def prepare_prediction_data(step, transaction_type, amount, oldbalance_org, 
                          newbalance_orig, oldbalance_dest, newbalance_dest):
    """
    Prepare input data for the pipeline prediction
    """
    # Encode transaction type using label encoder
    # Define the mapping used during training (adjust if your training used different encoding)
    type_encoding = {
        'CASH_IN': 0,
        'CASH_OUT': 1, 
        'DEBIT': 2,
        'PAYMENT': 3,
        'TRANSFER': 4
    }
    
    # Encode the transaction type
    encoded_type = type_encoding.get(transaction_type, 0)  # Default to 0 if unknown
    
    # Create a DataFrame with the exact column names expected by the pipeline
    data = pd.DataFrame({
        'step': [step],
        'type': [encoded_type],  # Use encoded integer value
        'amount': [amount],
        'oldbalanceOrg': [oldbalance_org],
        'newbalanceOrig': [newbalance_orig],
        'oldbalanceDest': [oldbalance_dest],
        'newbalanceDest': [newbalance_dest]
    })
    
    return data

def display_results(prediction, prediction_proba=None):
    """
    Display the prediction results in a nice format
    """
    st.markdown("---")
    st.subheader("🎯 Analysis Results")
    
    # Create columns for metrics
    if prediction_proba is not None:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
    
    # Main prediction result
    with col1:
        if prediction == 1:
            st.error("🚨 **FRAUD DETECTED**")
            st.markdown("**Status:** Fraudulent Transaction")
        else:
            st.success("✅ **LEGITIMATE**")
            st.markdown("**Status:** Normal Transaction")
    
    # Probability metrics (if available)
    if prediction_proba is not None:
        with col2:
            fraud_probability = prediction_proba[1] * 100
            st.metric(
                "Fraud Probability", 
                f"{fraud_probability:.1f}%",
                delta=None
            )
        
        with col3:
            confidence = max(prediction_proba) * 100
            st.metric(
                "Model Confidence", 
                f"{confidence:.1f}%",
                delta=None
            )
        
        # Risk level assessment
        st.markdown("### Risk Assessment")
        if fraud_probability >= 80:
            st.error("🔴 **HIGH RISK** - Immediate manual review required")
        elif fraud_probability >= 50:
            st.warning("🟡 **MEDIUM RISK** - Consider additional verification")
        elif fraud_probability >= 20:
            st.info("🟠 **LOW-MEDIUM RISK** - Monitor transaction")
        else:
            st.success("🟢 **LOW RISK** - Transaction appears normal")
        
        # Detailed probability breakdown
        with st.expander("Detailed Probability Breakdown"):
            prob_df = pd.DataFrame({
                'Class': ['Legitimate (0)', 'Fraud (1)'],
                'Probability': [f"{prediction_proba[0]:.4f}", f"{prediction_proba[1]:.4f}"],
                'Percentage': [f"{prediction_proba[0]*100:.2f}%", f"{prediction_proba[1]*100:.2f}%"]
            })
            st.dataframe(prob_df, use_container_width=True)
    
    else:
        with col2:
            st.info("Probability scores not available for this model")

if __name__ == "__main__":
    main()
