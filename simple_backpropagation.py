from manim import *

class SimpleBackpropagation(Scene):
    def construct(self):
        # Define the base pause time variable
        base_pause_time = 2

        # Title
        title = Text("Simple Backpropagation Example").scale(0.8)
        self.play(Write(title))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding
        self.play(FadeOut(title))

        # Step 1: Initialize the network with random weights
        input_neuron = Circle(radius=0.4, color=BLUE).move_to(LEFT * 5)
        hidden_neuron = Circle(radius=0.4, color=GREEN).move_to(ORIGIN)
        output_neuron = Circle(radius=0.4, color=RED).move_to(RIGHT * 5)

        input_label = Text("Input").scale(0.5).next_to(input_neuron, LEFT)
        hidden_label = Text("Hidden").scale(0.5).next_to(hidden_neuron, UP)
        output_label = Text("Output").scale(0.5).next_to(output_neuron, RIGHT)

        self.play(Write(input_neuron), Write(hidden_neuron), Write(output_neuron))
        self.play(Write(input_label), Write(hidden_label), Write(output_label))

        input_to_hidden = Arrow(input_neuron.get_right(), hidden_neuron.get_left(), buff=0.2)
        hidden_to_output = Arrow(hidden_neuron.get_right(), output_neuron.get_left(), buff=0.2)

        self.play(Write(input_to_hidden), Write(hidden_to_output))

        input_to_hidden_weight = Text("w1 = 0.5").scale(0.5).next_to(input_to_hidden, UP)
        hidden_to_output_weight = Text("w2 = -0.3").scale(0.5).next_to(hidden_to_output, UP)

        self.play(Write(input_to_hidden_weight), Write(hidden_to_output_weight))
        self.wait(base_pause_time)  # Extended pause for better understanding

        # Step 2: Forward pass
        input_value = Text("x = 1.0").scale(0.5).next_to(input_neuron, DOWN)
        self.play(Write(input_value))
        self.wait(base_pause_time)  # Extended pause for better understanding

        hidden_value = Text("h = sigmoid(0.5)").scale(0.4).next_to(hidden_neuron, DOWN)
        self.play(Write(hidden_value))
        self.wait(base_pause_time)  # Extended pause for better understanding

        output_expression_1 = Text("y = sigmoid(").scale(0.38).next_to(output_neuron, DOWN)
        output_expression_2 = Text("-0.3 * sigmoid(0.5)").scale(0.38).next_to(output_expression_1, RIGHT)
        output_expression_3 = Text(")").scale(0.38).next_to(output_expression_2, RIGHT)
        
        # Adjusting the positioning to fit the screen
        expression_group = VGroup(output_expression_1, output_expression_2, output_expression_3).arrange(RIGHT, buff=0.1).next_to(output_neuron, DOWN)

        self.play(Write(output_expression_1))
        self.play(Write(output_expression_2))
        self.play(Write(output_expression_3))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding

        # Clear forward pass text
        self.play(FadeOut(input_value), FadeOut(hidden_value), FadeOut(expression_group))

        # Step 3: Backward pass
        error = Text("Error = (target - y)").scale(0.5).next_to(output_neuron, DOWN)
        self.play(Write(error))
        self.wait(base_pause_time)  # Extended pause for better understanding

        self.play(FadeOut(error))
        output_gradient = Text("dE/dy").scale(0.5).next_to(output_neuron, DOWN)
        self.play(Write(output_gradient))
        self.wait(base_pause_time)  # Extended pause for better understanding

        self.play(FadeOut(output_gradient))
        hidden_gradient = Text("dE/dh").scale(0.5).next_to(hidden_neuron, DOWN)
        self.play(Write(hidden_gradient))
        self.wait(base_pause_time)  # Extended pause for better understanding

        self.play(FadeOut(hidden_gradient))

        # Step 4: Update weights
        update_w2 = Text("Δw2 = -learning_rate * dE/dw2").scale(0.5).next_to(hidden_to_output, DOWN * 2)
        self.play(Write(update_w2))
        self.wait(base_pause_time)  # Extended pause for better understanding

        new_w2 = Text("w2' = w2 + Δw2").scale(0.5).next_to(hidden_to_output_weight, DOWN * 2)
        self.play(Write(new_w2))
        self.wait(base_pause_time)  # Extended pause for better understanding

        self.play(FadeOut(update_w2))
        update_w1 = Text("Δw1 = -learning_rate * dE/dw1").scale(0.5).next_to(input_to_hidden, DOWN * 2)
        self.play(Write(update_w1))
        self.wait(base_pause_time)  # Extended pause for better understanding

        new_w1 = Text("w1' = w1 + Δw1").scale(0.5).next_to(input_to_hidden_weight, DOWN * 2)
        self.play(Write(new_w1))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding

        # Clear update weights text
        self.play(FadeOut(new_w2), FadeOut(update_w1), FadeOut(new_w1))

        # Final updated weights
        updated_weights_title = Text("Updated Weights:").scale(0.8).move_to(UP * 3)
        updated_w1 = Text("w1' = 0.55").scale(0.5).next_to(updated_weights_title, DOWN)
        updated_w2 = Text("w2' = -0.28").scale(0.5).next_to(updated_w1, DOWN)

        self.play(Write(updated_weights_title), Write(updated_w1), Write(updated_w2))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding

        # Summarize
        summary = Text("Repeat until error is minimized!").scale(0.8).to_edge(DOWN)
        self.play(Write(summary))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding

        # Clear screen before credits
        all_elements = VGroup(input_neuron, hidden_neuron, output_neuron, input_label, hidden_label, output_label, 
                              input_to_hidden, hidden_to_output, input_to_hidden_weight, hidden_to_output_weight,
                              updated_weights_title, updated_w1, updated_w2, summary)
        self.play(FadeOut(all_elements))
        #self.wait(base_pause_time * 5)  # Pause before displaying credits

        # Credits
        credits = Text("Created by Luís Cunha using Manim Library*\n* Developed by Grant Sanderson (3Blue1Brown)").scale(0.3).to_edge(DOWN)
        self.play(Write(credits))
        self.wait(base_pause_time * 2)  # Extended pause for better understanding

if __name__ == "__main__":
    from manim import *
    config.background_color = BLACK
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 30
    scene = SimpleBackpropagation()
    scene.render()
