#!/bin/bash

# Test script for print_transcript.py
# Tests Nova batch, Nova streaming, and Flux formats with various options

set -e  # Exit on error

SCRIPT="print_transcript.py"
TEST_DIR="test_data"
RESULTS_DIR="test_results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create test directories
mkdir -p "$TEST_DIR"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}Creating test data files...${NC}"

# =============================================================================
# Test Data: Nova Batch/Prerecorded
# =============================================================================

cat > "$TEST_DIR/nova_batch.json" << 'EOF'
{
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "transcript": "Hi. My name is Jason and I work at Deepgram.",
            "confidence": 0.95,
            "words": [
              {
                "word": "hi",
                "start": 0.64,
                "end": 0.96,
                "confidence": 0.98,
                "punctuated_word": "Hi."
              },
              {
                "word": "my",
                "start": 1.2,
                "end": 1.36,
                "confidence": 0.99,
                "punctuated_word": "My"
              },
              {
                "word": "name",
                "start": 1.36,
                "end": 1.52,
                "confidence": 0.99,
                "punctuated_word": "name"
              },
              {
                "word": "is",
                "start": 1.52,
                "end": 1.68,
                "confidence": 0.98,
                "punctuated_word": "is"
              },
              {
                "word": "jason",
                "start": 1.68,
                "end": 2.08,
                "confidence": 0.95,
                "punctuated_word": "Jason"
              },
              {
                "word": "and",
                "start": 2.24,
                "end": 2.4,
                "confidence": 0.99,
                "punctuated_word": "and"
              },
              {
                "word": "i",
                "start": 2.4,
                "end": 2.48,
                "confidence": 0.98,
                "punctuated_word": "I"
              },
              {
                "word": "work",
                "start": 2.48,
                "end": 2.72,
                "confidence": 0.97,
                "punctuated_word": "work"
              },
              {
                "word": "at",
                "start": 2.72,
                "end": 2.88,
                "confidence": 0.99,
                "punctuated_word": "at"
              },
              {
                "word": "deepgram",
                "start": 2.88,
                "end": 3.44,
                "confidence": 0.92,
                "punctuated_word": "Deepgram."
              }
            ]
          }
        ]
      }
    ]
  }
}
EOF

# =============================================================================
# Test Data: Nova Batch with Multiple Channels and Speakers
# =============================================================================

cat > "$TEST_DIR/nova_batch_multichannel.json" << 'EOF'
{
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "transcript": "Hello from channel one.",
            "confidence": 0.96,
            "words": [
              {
                "word": "hello",
                "start": 0.5,
                "end": 0.9,
                "confidence": 0.97,
                "punctuated_word": "Hello",
                "speaker": 0
              },
              {
                "word": "from",
                "start": 0.9,
                "end": 1.1,
                "confidence": 0.98,
                "punctuated_word": "from",
                "speaker": 0
              },
              {
                "word": "channel",
                "start": 1.1,
                "end": 1.5,
                "confidence": 0.96,
                "punctuated_word": "channel",
                "speaker": 0
              },
              {
                "word": "one",
                "start": 1.5,
                "end": 1.9,
                "confidence": 0.95,
                "punctuated_word": "one.",
                "speaker": 0
              }
            ]
          }
        ]
      },
      {
        "alternatives": [
          {
            "transcript": "Hello from channel two.",
            "confidence": 0.94,
            "words": [
              {
                "word": "hello",
                "start": 0.6,
                "end": 1.0,
                "confidence": 0.95,
                "punctuated_word": "Hello",
                "speaker": 1
              },
              {
                "word": "from",
                "start": 1.0,
                "end": 1.2,
                "confidence": 0.97,
                "punctuated_word": "from",
                "speaker": 1
              },
              {
                "word": "channel",
                "start": 1.2,
                "end": 1.6,
                "confidence": 0.93,
                "punctuated_word": "channel",
                "speaker": 1
              },
              {
                "word": "two",
                "start": 1.6,
                "end": 2.0,
                "confidence": 0.92,
                "punctuated_word": "two.",
                "speaker": 1
              }
            ]
          }
        ]
      }
    ]
  }
}
EOF

# =============================================================================
# Test Data: Nova Batch with Entities
# =============================================================================

cat > "$TEST_DIR/nova_batch_entities.json" << 'EOF'
{
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "transcript": "My phone number is 555-1234 and I live in San Francisco.",
            "confidence": 0.94,
            "words": [
              {"word": "my", "start": 0.0, "end": 0.2, "confidence": 0.99, "punctuated_word": "My"},
              {"word": "phone", "start": 0.2, "end": 0.5, "confidence": 0.98, "punctuated_word": "phone"},
              {"word": "number", "start": 0.5, "end": 0.8, "confidence": 0.97, "punctuated_word": "number"},
              {"word": "is", "start": 0.8, "end": 0.95, "confidence": 0.99, "punctuated_word": "is"},
              {"word": "555-1234", "start": 0.95, "end": 1.6, "confidence": 0.92, "punctuated_word": "555-1234"},
              {"word": "and", "start": 1.7, "end": 1.85, "confidence": 0.98, "punctuated_word": "and"},
              {"word": "i", "start": 1.85, "end": 1.95, "confidence": 0.99, "punctuated_word": "I"},
              {"word": "live", "start": 1.95, "end": 2.2, "confidence": 0.97, "punctuated_word": "live"},
              {"word": "in", "start": 2.2, "end": 2.35, "confidence": 0.99, "punctuated_word": "in"},
              {"word": "san", "start": 2.35, "end": 2.6, "confidence": 0.95, "punctuated_word": "San"},
              {"word": "francisco", "start": 2.6, "end": 3.2, "confidence": 0.94, "punctuated_word": "Francisco."}
            ],
            "entities": [
              {
                "label": "PHONE_NUMBER",
                "start_word": 4,
                "end_word": 5
              },
              {
                "label": "LOCATION",
                "start_word": 9,
                "end_word": 11
              }
            ]
          }
        ]
      }
    ]
  }
}
EOF

# =============================================================================
# Test Data: Nova Streaming
# =============================================================================

cat > "$TEST_DIR/nova_streaming.json" << 'EOF'
[
  {
    "type": "OpenStream",
    "headers": [
      ["dg-request-id", "test-request-123"]
    ],
    "received": "2025-01-15T10:00:00.000000"
  },
  {
    "type": "Results",
    "channel_index": [0],
    "duration": 0.96,
    "start": 0.64,
    "is_final": false,
    "speech_final": false,
    "channel": {
      "alternatives": [
        {
          "transcript": "Hi.",
          "confidence": 0.47,
          "words": [
            {
              "word": "hi",
              "start": 0.64,
              "end": 0.96,
              "confidence": 0.47,
              "punctuated_word": "Hi."
            }
          ]
        }
      ]
    },
    "received": "2025-01-15T10:00:01.000000"
  },
  {
    "type": "Results",
    "channel_index": [0],
    "duration": 1.52,
    "start": 0.64,
    "is_final": false,
    "speech_final": false,
    "channel": {
      "alternatives": [
        {
          "transcript": "Hi. My name",
          "confidence": 0.85,
          "words": [
            {
              "word": "hi",
              "start": 0.64,
              "end": 0.96,
              "confidence": 0.98,
              "punctuated_word": "Hi."
            },
            {
              "word": "my",
              "start": 1.2,
              "end": 1.36,
              "confidence": 0.92,
              "punctuated_word": "My"
            },
            {
              "word": "name",
              "start": 1.36,
              "end": 1.52,
              "confidence": 0.65,
              "punctuated_word": "name"
            }
          ]
        }
      ]
    },
    "received": "2025-01-15T10:00:01.500000"
  },
  {
    "type": "Results",
    "channel_index": [0],
    "duration": 2.8,
    "start": 0.64,
    "is_final": true,
    "speech_final": true,
    "channel": {
      "alternatives": [
        {
          "transcript": "Hi. My name is Jason.",
          "confidence": 0.95,
          "words": [
            {
              "word": "hi",
              "start": 0.64,
              "end": 0.96,
              "confidence": 0.98,
              "punctuated_word": "Hi."
            },
            {
              "word": "my",
              "start": 1.2,
              "end": 1.36,
              "confidence": 0.99,
              "punctuated_word": "My"
            },
            {
              "word": "name",
              "start": 1.36,
              "end": 1.52,
              "confidence": 0.99,
              "punctuated_word": "name"
            },
            {
              "word": "is",
              "start": 1.52,
              "end": 1.68,
              "confidence": 0.98,
              "punctuated_word": "is"
            },
            {
              "word": "jason",
              "start": 1.68,
              "end": 2.08,
              "confidence": 0.95,
              "punctuated_word": "Jason."
            }
          ]
        }
      ]
    },
    "received": "2025-01-15T10:00:02.100000"
  },
  {
    "type": "UtteranceEnd",
    "channel": [0],
    "last_word_end": 2.08,
    "received": "2025-01-15T10:00:03.000000"
  },
  {
    "type": "Metadata",
    "request_id": "test-request-123",
    "duration": 3.5,
    "channels": 1,
    "received": "2025-01-15T10:00:04.000000"
  }
]
EOF

# =============================================================================
# Test Data: Flux Streaming
# =============================================================================

cat > "$TEST_DIR/flux_streaming.json" << 'EOF'
[
  {
    "type": "OpenStream",
    "headers": [
      ["dg-request-id", "flux-request-456"]
    ],
    "received": "2025-01-15T11:00:00.000000"
  },
  {
    "type": "Connected",
    "request_id": "flux-request-456",
    "sequence_id": 0,
    "received": "2025-01-15T11:00:00.100000"
  },
  {
    "type": "TurnInfo",
    "event": "Update",
    "turn_index": 0,
    "audio_window_start": 0.0,
    "audio_window_end": 1.2,
    "transcript": "Hi,",
    "words": [
      {
        "word": "Hi,",
        "confidence": 0.85
      }
    ],
    "end_of_turn_confidence": 0.02,
    "received": "2025-01-15T11:00:01.200000"
  },
  {
    "type": "TurnInfo",
    "event": "StartOfTurn",
    "turn_index": 0,
    "audio_window_start": 0.0,
    "audio_window_end": 1.2,
    "transcript": "Hi,",
    "words": [
      {
        "word": "Hi,",
        "confidence": 0.85
      }
    ],
    "end_of_turn_confidence": 0.02,
    "received": "2025-01-15T11:00:01.250000"
  },
  {
    "type": "TurnInfo",
    "event": "Update",
    "turn_index": 0,
    "audio_window_start": 0.0,
    "audio_window_end": 2.5,
    "transcript": "Hi. My name is",
    "words": [
      {
        "word": "Hi.",
        "confidence": 0.95
      },
      {
        "word": "My",
        "confidence": 0.93
      },
      {
        "word": "name",
        "confidence": 0.91
      },
      {
        "word": "is",
        "confidence": 0.94
      }
    ],
    "end_of_turn_confidence": 0.15,
    "received": "2025-01-15T11:00:02.500000"
  },
  {
    "type": "TurnInfo",
    "event": "EndOfTurn",
    "turn_index": 0,
    "audio_window_start": 0.0,
    "audio_window_end": 3.8,
    "transcript": "Hi. My name is Jason and I work at Deepgram.",
    "words": [
      {
        "word": "Hi.",
        "confidence": 0.98
      },
      {
        "word": "My",
        "confidence": 0.97
      },
      {
        "word": "name",
        "confidence": 0.96
      },
      {
        "word": "is",
        "confidence": 0.98
      },
      {
        "word": "Jason",
        "confidence": 0.94
      },
      {
        "word": "and",
        "confidence": 0.97
      },
      {
        "word": "I",
        "confidence": 0.98
      },
      {
        "word": "work",
        "confidence": 0.96
      },
      {
        "word": "at",
        "confidence": 0.99
      },
      {
        "word": "Deepgram.",
        "confidence": 0.93
      }
    ],
    "end_of_turn_confidence": 0.92,
    "received": "2025-01-15T11:00:03.800000"
  }
]
EOF

echo -e "${GREEN}✓ Test data files created${NC}\n"

# =============================================================================
# Test Functions
# =============================================================================

run_test() {
    local test_name="$1"
    local input_file="$2"
    shift 2
    local options="$@"
    
    echo -e "${BLUE}Running: $test_name${NC}"
    echo "  Command: python $SCRIPT -f $input_file $options"
    
    output_file="$RESULTS_DIR/${test_name}.txt"
    
    if python "$SCRIPT" -f "$input_file" $options > "$output_file" 2>&1; then
        echo -e "${GREEN}  ✓ PASSED${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}  ✗ FAILED${NC}"
        echo "  See $output_file for details"
        echo ""
        return 1
    fi
}

# =============================================================================
# Run Tests
# =============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Nova Batch Format${NC}"
echo -e "${BLUE}========================================${NC}\n"

run_test "nova_batch_default" "$TEST_DIR/nova_batch.json"
run_test "nova_batch_only_transcript" "$TEST_DIR/nova_batch.json" "--only-transcript"
run_test "nova_batch_with_speakers" "$TEST_DIR/nova_batch.json" "--print-speakers"
run_test "nova_batch_colorized" "$TEST_DIR/nova_batch.json" "--colorize"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Nova Batch Multi-channel${NC}"
echo -e "${BLUE}========================================${NC}\n"

run_test "nova_multichannel_default" "$TEST_DIR/nova_batch_multichannel.json"
run_test "nova_multichannel_with_channels" "$TEST_DIR/nova_batch_multichannel.json" "--print-channels"
run_test "nova_multichannel_with_speakers" "$TEST_DIR/nova_batch_multichannel.json" "--print-speakers"
run_test "nova_multichannel_channels_and_speakers" "$TEST_DIR/nova_batch_multichannel.json" "--print-channels --print-speakers"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Nova Batch with Entities${NC}"
echo -e "${BLUE}========================================${NC}\n"

run_test "nova_entities_default" "$TEST_DIR/nova_batch_entities.json"
run_test "nova_entities_with_entities" "$TEST_DIR/nova_batch_entities.json" "--print-entities"
run_test "nova_entities_only_transcript" "$TEST_DIR/nova_batch_entities.json" "--print-entities --only-transcript"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Nova Streaming Format${NC}"
echo -e "${BLUE}========================================${NC}\n"

run_test "nova_streaming_default" "$TEST_DIR/nova_streaming.json"
run_test "nova_streaming_with_interim" "$TEST_DIR/nova_streaming.json" "--print-interim"
run_test "nova_streaming_with_received" "$TEST_DIR/nova_streaming.json" "--print-received"
run_test "nova_streaming_with_delay" "$TEST_DIR/nova_streaming.json" "--print-delay"
run_test "nova_streaming_all_options" "$TEST_DIR/nova_streaming.json" "--print-interim --print-received --print-delay --print-speakers"
run_test "nova_streaming_only_transcript" "$TEST_DIR/nova_streaming.json" "--only-transcript"
run_test "nova_streaming_only_transcript_with_interim" "$TEST_DIR/nova_streaming.json" "--only-transcript --print-interim"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Flux Streaming Format${NC}"
echo -e "${BLUE}========================================${NC}\n"

run_test "flux_streaming_default" "$TEST_DIR/flux_streaming.json"
run_test "flux_streaming_with_interim" "$TEST_DIR/flux_streaming.json" "--print-interim"
run_test "flux_streaming_with_received" "$TEST_DIR/flux_streaming.json" "--print-received"
run_test "flux_streaming_with_delay" "$TEST_DIR/flux_streaming.json" "--print-delay"
run_test "flux_streaming_all_options" "$TEST_DIR/flux_streaming.json" "--print-interim --print-received --print-delay"
run_test "flux_streaming_only_transcript" "$TEST_DIR/flux_streaming.json" "--only-transcript"
run_test "flux_streaming_colorized" "$TEST_DIR/flux_streaming.json" "--colorize"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}\n"

total_tests=$(ls -1 "$RESULTS_DIR" | wc -l)
echo -e "Total tests run: ${GREEN}$total_tests${NC}"
echo -e "\nTest results saved in: ${BLUE}$RESULTS_DIR${NC}"
echo -e "Test data saved in: ${BLUE}$TEST_DIR${NC}"

echo -e "\n${GREEN}All tests completed!${NC}"
echo -e "\nTo view a specific test result, use:"
echo -e "  cat $RESULTS_DIR/<test_name>.txt"
echo -e "\nTo view all results:"
echo -e "  ls $RESULTS_DIR"
