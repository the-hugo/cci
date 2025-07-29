if __name__ == '__main__':
    import cci
    print('Comparing original vs improved CCI scores...')

    # Load data
    df = cci.i_o.load_dialogues('cci/data/data.pkl')
    
    print('\n=== RUNNING IMPROVED VERSION ===')
    try:
        result_improved = cci.pipeline.compute_cci_with_components(df, use_multiprocessing=False)
        
        non_zero_improved = result_improved[result_improved['CCI_score'] != 0]
        
        print(f'\nIMPROVED Results:')
        print(f'Total speaker pairs: {len(result_improved):,}')
        print(f'Non-zero interactions: {len(non_zero_improved):,}')
        
        if len(non_zero_improved) > 0:
            print(f'CCI scores - Mean: {non_zero_improved["CCI_score"].mean():.6f}, Max: {non_zero_improved["CCI_score"].max():.6f}')
            print(f'Incorporation - Mean: {non_zero_improved["I_mean"].mean():.4f}, Max: {non_zero_improved["I_mean"].max():.4f}')
            print(f'Shared Growth - Mean: {non_zero_improved["S_mean"].mean():.4f}')
            
            # Show configuration
            print(f'\nImprovement configuration:')
            print(f'  - Lookahead window: {cci.config.LOOKAHEAD_WINDOW} turns')
            print(f'  - Semantic threshold: {cci.config.SEMANTIC_THRESHOLD}')
            print(f'  - Alpha (incorporation weight): {cci.config.ALPHA}')
        else:
            print('No non-zero scores found!')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback 
        traceback.print_exc()