import cv2
import mediapipe as mp
import numpy as np
import random
import time

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.winner = None
        self.game_over = False
        self.scores = {'X': 0, 'O': 0, 'Tie': 0}
        self.play_against_computer = None

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.check_winner()
            self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                self.winner = self.board[combo[0]]
                self.game_over = True
                self.scores[self.winner] += 1
                return
        if ' ' not in self.board:
            self.game_over = True
            self.scores['Tie'] += 1

    def reset_game(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.winner = None
        self.game_over = False

    def computer_move(self):
        empty_positions = [i for i, spot in enumerate(self.board) if spot == ' ']
        return random.choice(empty_positions)

def draw_board(frame, game):
    h, w, _ = frame.shape
    cell_height, cell_width = h // 3, w // 3

    # Draw grid lines
    cv2.line(frame, (cell_width, 0), (cell_width, h), (255, 255, 255), 2)
    cv2.line(frame, (2 * cell_width, 0), (2 * cell_width, h), (255, 255, 255), 2)
    cv2.line(frame, (0, cell_height), (w, cell_height), (255, 255, 255), 2)
    cv2.line(frame, (0, 2 * cell_height), (w, 2 * cell_height), (255, 255, 255), 2)

    for i in range(9):
        row, col = i // 3, i % 3
        center = (col * cell_width + cell_width // 2, row * cell_height + cell_height // 2)
        if game.board[i] == 'X':
            cv2.line(frame, (center[0] - 30, center[1] - 30), (center[0] + 30, center[1] + 30), (0, 0, 255), 3)
            cv2.line(frame, (center[0] + 30, center[1] - 30), (center[0] - 30, center[1] + 30), (0, 0, 255), 3)
        elif game.board[i] == 'O':
            cv2.circle(frame, center, 30, (0, 255, 0), 3)

    # Display scores
    cv2.putText(frame, f"X: {game.scores['X']} O: {game.scores['O']} Tie: {game.scores['Tie']}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_selection_menu(frame):
    h, w, _ = frame.shape
    cv2.putText(frame, "Select game mode:", (w // 4, h // 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Play against computer", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Play against human", (w // 4, 2 * h // 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def get_selection(frame, index_finger_tip):
    h, w, _ = frame.shape
    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
    
    if h // 3 < y < 2 * h // 3:
        return True  # Play against computer
    elif 2 * h // 3 < y < h:
        return False  # Play against human
    return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    game = TicTacToe()

    selection_start_time = None
    selection_duration = 2  # Hold for 2 seconds to make a selection

    while game.play_against_computer is None:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_selection_menu(frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            selection = get_selection(frame, index_finger_tip)
            if selection is not None:
                if selection_start_time is None:
                    selection_start_time = time.time()
                elif time.time() - selection_start_time >= selection_duration:
                    game.play_against_computer = selection
            else:
                selection_start_time = None

            # Draw a circle at the index finger tip
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        cv2.imshow("TicTacToe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_board(frame, game)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if hand is closed (thumb tip close to index finger tip)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                if distance < 0.1:  # Adjust this threshold as needed
                    h, w, _ = frame.shape
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    cell_height, cell_width = h // 3, w // 3
                    row, col = y // cell_height, x // cell_width
                    position = row * 3 + col

                    if not game.game_over and game.board[position] == ' ':
                        game.make_move(position)
                        if game.play_against_computer and not game.game_over:
                            computer_position = game.computer_move()
                            game.make_move(computer_position)

        if game.game_over:
            if game.winner:
                cv2.putText(frame, f"Player {game.winner} wins!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "It's a tie!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("TicTacToe", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game.reset_game()
            game.play_against_computer = None
            selection_start_time = None
            while game.play_against_computer is None:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                draw_selection_menu(frame)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    selection = get_selection(frame, index_finger_tip)
                    if selection is not None:
                        if selection_start_time is None:
                            selection_start_time = time.time()
                        elif time.time() - selection_start_time >= selection_duration:
                            game.play_against_computer = selection
                    else:
                        selection_start_time = None

                    # Draw a circle at the index finger tip
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                cv2.imshow("TicTacToe", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()