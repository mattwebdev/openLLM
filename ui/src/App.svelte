<script>
  import { onMount } from 'svelte';
  import { Button, TextField } from '@smui/textfield';
  import { Card } from '@smui/card';

  let input = '';
  let output = '';
  let loading = false;
  let error = null;

  async function handleSubmit() {
    loading = true;
    error = null;
    try {
      const response = await fetch('http://localhost:8080/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate response');
      }

      const data = await response.json();
      output = data.output;
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }
</script>

<main>
  <Card>
    <div class="container">
      <h1>OpenLLM Test Interface</h1>
      
      <div class="input-section">
        <TextField
          label="Input Text"
          bind:value={input}
          fullWidth
          multiline
          rows={4}
        />
        <Button
          on:click={handleSubmit}
          disabled={loading || !input.trim()}
        >
          {loading ? 'Generating...' : 'Generate'}
        </Button>
      </div>

      {#if error}
        <div class="error">
          {error}
        </div>
      {/if}

      {#if output}
        <div class="output-section">
          <h2>Generated Output</h2>
          <div class="output">
            {output}
          </div>
        </div>
      {/if}
    </div>
  </Card>
</main>

<style>
  main {
    padding: 2rem;
    max-width: 800px;
    margin: 0 auto;
  }

  .container {
    padding: 1rem;
  }

  h1 {
    margin-bottom: 2rem;
    text-align: center;
  }

  .input-section {
    margin-bottom: 2rem;
  }

  .output-section {
    margin-top: 2rem;
  }

  .output {
    padding: 1rem;
    background: #f5f5f5;
    border-radius: 4px;
    white-space: pre-wrap;
  }

  .error {
    color: red;
    margin: 1rem 0;
  }
</style> 