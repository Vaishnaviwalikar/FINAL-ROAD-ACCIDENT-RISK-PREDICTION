def load_model_and_preprocessing():
    """Load the trained model, label encoders, and scaler."""
    # Declare all global variables at the beginning
    global a_model, a_feature_columns, a_label_encoders, a_config, a_processed_dir, a_scaler, a_checkpoint_path, a_device

    # Force CPU for all operations
    a_device = 'cpu'

    # Initialize all global variables
    a_model = None
    a_feature_columns = []
    a_label_encoders = {}
    a_config = {}
    a_processed_dir = ""
    a_scaler = None
    a_checkpoint_path = None

    # Set up logging
    logger = setup_logging()

    # Initialize logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize globals if not already set
    if a_feature_columns is None:
        a_feature_columns = []

    try:
        logger.info("\n" + "="*70)
        logger.info("🚀 Starting RoadSafe AI (Hybrid Mode)")
        logger.info("="*70 + "\n")
        logger.info("🔍 Initializing Model Loading Process...")

        # Try to import torch first
        torch = None
        try:
            import torch
            logger.info("✅ PyTorch imported successfully")
        except ImportError:
            logger.warning("⚠️ PyTorch not available - will use fallback mode")
            return

        # Try to import other ML libraries
        try:
            import pandas as pd
            from joblib import load as joblib_load
            import shap
            import os
            logger.info("✅ All ML dependencies available")
        except ImportError as e:
            logger.error(f"⚠️ Some ML dependencies not available: {e}")
            logger.warning("🔄 Running in demo mode with fallback predictions")
            return

        # Always use CPU to avoid device mismatch issues
        a_device = torch.device("cpu")
        logger.info(f"🎯 Forcing device to: {a_device} (CPU only mode)")

        # Disable CUDA
        torch.cuda.is_available = lambda: False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Try to load the model from quick_fixed directory
        model_path = os.path.join("outputs", "quick_fixed", "best.pt")
        if os.path.exists(model_path):
            logger.info(f"🔍 Found model at: {os.path.abspath(model_path)}")
            try:
                # Load the checkpoint with CPU mapping
                logger.info("🔄 Loading model checkpoint with CPU mapping...")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                logger.info("✅ Successfully loaded model checkpoint")
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

                # Set feature columns from checkpoint or use defaults
                a_feature_columns = checkpoint.get('features', [
                    'hour', 'Day_of_Week', 'month', 'Latitude', 'Longitude',
                    'Speed_limit', 'Light_Conditions', 'Road_Surface_Conditions',
                    'Weather_Conditions', 'Road_Type', 'Junction_Detail',
                    'Number_of_Vehicles', 'Number_of_Casualties'
                ])

                # Store config
                a_config = checkpoint.get('cfg', {})

                logger.info(f"✅ Model loaded successfully with {len(a_feature_columns)} features")
                logger.info(f"Model device: {a_device}")
                logger.info("="*70 + "\n")

            except Exception as e:
                logger.error(f"❌ Error initializing model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                a_model = None
                a_feature_columns = []
        else:
            logger.error(f"❌ Model file not found at: {os.path.abspath(model_path)}")

    except Exception as e:
        logger.critical(f"\n❌❌ Critical error in load_model_and_preprocessing: {str(e)}")
        a_model = None
        a_feature_columns = []

        # Log traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        a_feature_columns = None
