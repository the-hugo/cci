if __name__ == '__main__':
    import cci
    print('Testing improvements...')

    # Load data
    df = cci.i_o.load_dialogues('cci/data/data.pkl')

    # Test just a few dialogues first
    test_df = df[df['dialogue_id'].isin(df['dialogue_id'].unique()[:3])].copy()
    print(f'Testing with {len(test_df)} turns from 3 dialogues...')

    # Run component analysis with improvements
    try:
        result = cci.pipeline.compute_cci_with_components(test_df, use_multiprocessing=False)
        
        non_zero = result[result['CCI_score'] != 0]
        if len(non_zero) > 0:
            print(f'\nIMPROVED Results (3 dialogues):')
            print(f'Non-zero interactions: {len(non_zero)}')
            print(f'D_t (Divergence): {non_zero["D_mean"].mean():.4f}')
            print(f'I_t+1 (Incorporation): {non_zero["I_mean"].mean():.4f}')
            print(f'S_t+1 (Shared Growth): {non_zero["S_mean"].mean():.4f}')
            print(f'CCI score: {non_zero["CCI_score"].mean():.6f}')
            
            # Show improvement details
            print(f'\nIncorporation rate improvements:')
            print(f'  - Extended lookahead: 4 → 8 turns')
            print(f'  - Added semantic matching with threshold: {cci.config.SEMANTIC_THRESHOLD}')
            
        else:
            print('No non-zero CCI scores in test subset')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()