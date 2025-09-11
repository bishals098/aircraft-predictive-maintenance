import pandas as pd
import numpy as np
import os
import datetime

class NASADataLoader:
    def __init__(self):
        self.column_names = [
            'unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)]
        
        # Dataset characteristics
        self.dataset_info = {
            'FD001': {'fault_modes': 1, 'operating_conditions': 1, 'train_engines': 100, 'test_engines': 100},
            'FD002': {'fault_modes': 1, 'operating_conditions': 6, 'train_engines': 260, 'test_engines': 259},
            'FD003': {'fault_modes': 2, 'operating_conditions': 1, 'train_engines': 100, 'test_engines': 100},
            'FD004': {'fault_modes': 2, 'operating_conditions': 6, 'train_engines': 249, 'test_engines': 248}
        }

    def load_nasa_dataset(self, train_file, test_file, rul_file=None):
        """Load NASA C-MAPSS dataset with enhanced error handling"""
        try:
            # Load training data
            train_df = pd.read_csv(train_file, sep=' ', header=None, 
                                 names=self.column_names, index_col=False)
            train_df.dropna(axis=1, inplace=True)
            
            # Load test data
            test_df = pd.read_csv(test_file, sep=' ', header=None,
                                names=self.column_names, index_col=False)
            test_df.dropna(axis=1, inplace=True)
            
            # Calculate RUL for training data
            train_df = self.calculate_rul(train_df)
            
            # Load true RUL for test data if available
            if rul_file and os.path.exists(rul_file):
                rul_df = pd.read_csv(rul_file, header=None, names=['true_rul'])
                
                # Add RUL to test data
                test_grouped = test_df.groupby('unit_id')['time_cycles'].max().reset_index()
                test_grouped['true_rul'] = rul_df['true_rul'].values
                test_df = test_df.merge(test_grouped[['unit_id', 'true_rul']], on='unit_id')
                
                # Calculate current RUL for test data
                test_df['remaining_useful_life'] = (
                    test_df['true_rul'] - 
                    (test_df.groupby('unit_id')['time_cycles'].transform('max') - test_df['time_cycles'])
                )
            else:
                # Fallback RUL calculation for test data
                test_df = self.calculate_rul(test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def load_all_datasets(self, nasa_data_path='data/nasa/'):
        """Load all NASA C-MAPSS datasets (FD001-FD004)"""
        all_datasets = {}
        
        for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
            print(f"\nLoading NASA {dataset} dataset...")
            
            train_file = f"{nasa_data_path}/train_{dataset}.txt"
            test_file = f"{nasa_data_path}/test_{dataset}.txt"
            rul_file = f"{nasa_data_path}/RUL_{dataset}.txt"
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [train_file, test_file, rul_file]):
                print(f"❌ {dataset} files not found, skipping...")
                continue
            
            try:
                train_df, test_df = self.load_nasa_dataset(train_file, test_file, rul_file)
                
                # Add dataset identifier
                train_df['dataset'] = dataset
                test_df['dataset'] = dataset
                
                # Create failure labels with dataset-specific thresholds
                threshold = self.get_failure_threshold(dataset)
                train_df = self.create_failure_labels(train_df, threshold=threshold)
                test_df = self.create_failure_labels(test_df, threshold=threshold)
                
                # Map to project format
                train_df = self.map_to_project_format(train_df)
                test_df = self.map_to_project_format(test_df)
                
                # Combine for processing
                combined_df = pd.concat([train_df, test_df], ignore_index=True)
                
                # Save processed data
                os.makedirs('data/processed', exist_ok=True)
                combined_df.to_csv(f'data/processed/nasa_aircraft_sensor_data_{dataset}.csv', index=False)
                
                all_datasets[dataset] = {
                    'combined_df': combined_df,
                    'train_df': train_df,
                    'test_df': test_df,
                    'info': self.dataset_info[dataset]
                }
                
                print(f"✅ {dataset}: {len(combined_df)} samples, "
                      f"{combined_df['unit_id'].nunique()} engines, "
                      f"Fault modes: {self.dataset_info[dataset]['fault_modes']}, "
                      f"Operating conditions: {self.dataset_info[dataset]['operating_conditions']}")
                
            except Exception as e:
                print(f"❌ Error loading {dataset}: {e}")
                continue
        
        return all_datasets

    def get_failure_threshold(self, dataset):
        """Get dataset-specific failure thresholds"""
        thresholds = {
            'FD001': 30,  # Single fault, single condition
            'FD002': 25,  # Single fault, multiple conditions (more challenging)
            'FD003': 30,  # Multiple faults, single condition
            'FD004': 20   # Multiple faults, multiple conditions (most challenging)
        }
        return thresholds.get(dataset, 30)

    def calculate_rul(self, df):
        """Calculate Remaining Useful Life for each engine"""
        max_cycles = df.groupby('unit_id')['time_cycles'].max()
        df['remaining_useful_life'] = df.apply(
            lambda row: max_cycles[row['unit_id']] - row['time_cycles'], axis=1
        )
        return df

    def create_failure_labels(self, df, threshold=30):
        """Create binary failure risk labels"""
        df['failure_risk'] = (df['remaining_useful_life'] <= threshold).astype(int)
        return df
    
    def map_to_project_format(self, df):
        """Map NASA sensor columns to project's sensor names"""
        sensor_mapping = {
            'sensor_2': 'temperature',    # Total temperature at fan inlet
            'sensor_3': 'pressure',       # Total pressure at fan inlet  
            'sensor_7': 'vibration',      # Static pressure at HPC outlet
            'sensor_9': 'rpm',           # Physical fan speed
            'sensor_13': 'oil_level',     # Corrected fan speed
            'sensor_17': 'fuel_flow',     # Fuel flow
            'op_setting_3': 'altitude',   # Altitude (operational setting)
            'sensor_12': 'speed'          # Static pressure at bypass-duct
        }
    
        # Create mapped columns
        for nasa_col, project_col in sensor_mapping.items():
            if nasa_col in df.columns:
                df[project_col] = df[nasa_col]
    
        # Add additional columns
        df['aircraft_id'] = df['dataset'] + '_' + df['unit_id'].astype(str)
        df['component_id'] = 'ENG_' + df['unit_id'].astype(str)
        df['timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df['time_cycles'], unit='D')
    
        # OPTION 1: Remove dataset column entirely (recommended)
        # df = df.drop('dataset', axis=1)
    
        # OPTION 2: Convert dataset to numeric encoding (if you need to keep it)
        dataset_encoding = {'FD001': 1, 'FD002': 2, 'FD003': 3, 'FD004': 4}
        df['dataset_encoded'] = df['dataset'].map(dataset_encoding)
        df = df.drop('dataset', axis=1)  # Remove original string column
    
        return df

    def get_dataset_statistics(self, all_datasets):
        """Generate comprehensive statistics for all datasets"""
        stats = {}
        
        for dataset, data in all_datasets.items():
            df = data['combined_df']
            info = data['info']
            
            stats[dataset] = {
                'total_samples': len(df),
                'total_engines': df['unit_id'].nunique(),
                'avg_cycles_per_engine': df.groupby('unit_id')['time_cycles'].max().mean(),
                'failure_rate': df['failure_risk'].mean(),
                'avg_rul': df['remaining_useful_life'].mean(),
                'fault_modes': info['fault_modes'],
                'operating_conditions': info['operating_conditions'],
                'sensor_ranges': {
                    'temperature': (df['temperature'].min(), df['temperature'].max()),
                    'pressure': (df['pressure'].min(), df['pressure'].max()),
                    'vibration': (df['vibration'].min(), df['vibration'].max()),
                    'rpm': (df['rpm'].min(), df['rpm'].max())
                }
            }
        
        return stats