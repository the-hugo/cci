if __name__ == '__main__':
    import cci
    print('Quick test of improvements...')

    # Load data
    df = cci.i_o.load_dialogues('cci/data/data.pkl')

    # Test just one small dialogue
    first_dialogue = df['dialogue_id'].iloc[0]
    test_df = df[df['dialogue_id'] == first_dialogue].head(20).copy()  # Just first 20 turns
    print(f'Testing with {len(test_df)} turns from dialogue {first_dialogue}...')

    # Run component analysis with improvements
    try:
        result = cci.pipeline.compute_cci_with_components(test_df, use_multiprocessing=False)
        
        print(f'Total speaker pairs: {len(result)}')
        non_zero = result[result['CCI_score'] != 0]
        
        if len(non_zero) > 0:
            print(f'\nIMPROVED Results:')
            print(f'Non-zero interactions: {len(non_zero)}')
            print(f'I_t+1 (Incorporation): {non_zero["I_mean"].mean():.4f}')
            print(f'CCI score: {non_zero["CCI_score"].mean():.6f}')
            
            # Show top result
            top_result = non_zero.iloc[0]
            print(f'\nTop interaction: {top_result["from_speaker"]} → {top_result["to_speaker"]}')
            print(f'  CCI: {top_result["CCI_score"]:.6f}')
            print(f'  Incorporation: {top_result["I_mean"]:.4f}')
            
        else:
            print('No non-zero CCI scores found')
            print('This might indicate the improvements need more tuning')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()