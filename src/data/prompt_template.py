ANSWER_AUGMENT_TEMPLATE = """Your task is to solve a math word problem. You should solve the problem step by step. At the end of your solution, write the final answer in the form of "The answer is X". Here are two examples:

## Example 1

Question:
Let $F_1 = (0,1)$ and $F_ 2= (4,1).$  Then the set of points $P$ such that\n\\[PF_1 + PF_2 = 6\\]form an ellipse.  The equation of this ellipse can be written as\n\\[\\frac{(x - h)^2}{a^2} + \\frac{(y - k)^2}{b^2} = 1.\\]Find $h + k + a + b.$

Solution:
We have that $2a = 6,$ so $a = 3.$  The distance between the foci is $2c = 4,$ so $c = 2.$  Hence, $b = \\sqrt{a^2 - c^2} = \\sqrt{5}.$\n\nThe center of the ellipse is the midpoint of $\\overline{F_1 F_2},$ which is $(2,1).$  Thus, the equation of the ellipse is\n\\[\\frac{(x - 2)^2}{3^2} + \\frac{(y - 1)^2}{(\\sqrt{5})^2} = 1.\\]Hence, $h + k + a + b = 2 + 1 + 3 + \\sqrt{5} = \\boxed{6 + \\sqrt{5}}.$. The answer is 6+\\sqrt{5}.

## Example 2

Question:
Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. If there are 6 jaguars in a forest, how many beetles are eaten each day?

Solution:
First find the total number of snakes eaten: 5 snakes/jaguar * 6 jaguars = 30 snakes\nThen find the total number of birds eaten per day: 30 snakes * 3 birds/snake = 90 snakes\nThen multiply the number of snakes by the number of beetles per snake to find the total number of beetles eaten per day: 90 snakes * 12 beetles/snake = 1080 beetles\nThe answer is 1080.

Now solve the following problem. The solution must end with "The answer is XXX" where XXX should be the final answer to the question.

Question:
$$QUESTION$$

Solution:"""


# Adapted from "Common 7B Language Models Already Possess Strong Math Capabilities"
QUESTION_AUGMENT_TEMPLATE = """Please act as a professional math teacher. Your goal is to create high quality math problems to help students learn math. You will be given a math question. Please generate a similar but new question according to the Given Question.

You have four principles to do this.
# Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and have a definite answer. For example, DO NOT ask, "what is the amount of A, B and C?".
# Ensure the new question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer.
# Ensure your student can answer the new question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your new question.
# Ensure your created question is solvable. Write the solution to it after the question.

Given Question: $$QUESTION$$

Now write a new question and its solution. The question must begin with "New Question:" and the solution must begin with "Solution to the New Question:". The solution must end with "The answer is XXX" where XXX should be the final answer to the question."""


ALTERNATIVE_FOLLOWUP_REFLECT_TEMPLATE = """You are a professional math teacher, and your goal is to teach your student to learn a given math problem. Now that your student has successfully solved the original problem, in order to make the student thoroughly understand the involved knowledge and problem-solving methodology, your task is to write a reflection section that go through the problem-solving process and provide additional insights. The reflection section should include the following components:

1. **Alternative Reasoning**: Present an alternative approach to solve the original problem. This alternative approach should be distinct from the original solution and still lead to the correct answer. While writing the alternative reasoning approach, consider explaining the principle of the methodology used in the original solution, how the alternative approach differs from the original method, and why it leads to the same correct answer.

2. **Follow-up Reasoning**: Associate the solution to a broader class of problems. You can either create a general form of the original problem to encourage the student to reduce reliance on specific values (e.g., use letters or variables to replace specific numbers in the original problem), or apply the concepts and methodologies from the original problem to a more challenging situation. Please do not just replace the original numbers in the question with new numbers, because that is essentially the same problem. The follow-up problem must also be solvable, and you need to provide the solution for it. Besides, please explain briefly how the new scenario associates with the original problem.

Example 1:

Original Problem:
Youngsville had a population of 684 people. The town had a growth spurt and the population increased by 25% then they witnessed that 40% of the population moved away. What is the current population?

Solution to the Original Problem:
The town had 684 people, and then had a 25% growth spurt, so the population increased by 684*0.25 = 171 people. This increase brought the population to 684+171 = 855 people. 40% of the population moved away, so 855*0.40 = 342 people moved away. The new population is 855-342 = 513 people. The answer is 513.

Alternative Reasoning:
The key to solve the problem is to understand the concept of relative increase and decrease percentages. Increasing by a% means the population grows to (100+a)% of the original, while decreasing by b% means the population reduces to (100-b)% based on the increased population. Therefore, this is essentially a problem of consecutive multiplication: multiply the initial total population by the percentage of change twice.
Therefore, an alternative calculation involves deriving a single effective percentage change of the whole process. A 25% increase is equivalent to multiplying by 1.25, and a 40% decrease is equivalent to multiplying by 0.60. Combining these two changes, the effective percentage change is 1.25 * 0.60 = 0.75, which corresponds to a 25% decrease from the original population. Therefore, the current population is 684 * 0.75 = 513. The alternative approach leads to the same result because the associative property of multiplication: (684 * 1.25) * 0.60 = 684 * (1.25 * 0.60) = 684 * 0.75 = 513.

Follow-up reasoning:
Let's think of a more general scenario. Suppose a town has a population of $P$ people. The population increases by $a$ percent, then $b$ percent of the population moves away, and we would like to know the final population. In this context, the first increase corresponds to multiplying by $(1 + a/100)$, and the subsequent decrease corresponds to multiplying by $(1 - b/100)$. So the total population change is $(1 + a/100) * (1 - b/100)$. Therefore, the final population is $P * (1 + a/100) * (1 - b/100)$. This abstract problem allows us to apply the same principles of relative percentage changes to calculate the final population based on the initial population and the two percentage changes. This generalization helps to understand the problem conceptually and apply it to various scenarios.

Example 2:

Original Problem:
Solve the equation (x-99)(x-101)=8.

Solution to the Original Problem:
Let t=x-100. Then the equation becomes (t-1)(t+1)=8, which transforms into t^2-1=8. Therefore, t=3 or t=-3, and accordingly we get x=97 or x=103. The answer is 97 or 103.

Alternative Reasoning:
The essence of substitution is to identify and simplify the common components of variable expressions by introducing a new variable, thereby reducing the complexity. Let's revisit the original equation. Expressions x-99 and x-101 share a similar form: a large constant offset from x. Due to the minimal difference between 99 and 101, we can use substitution to transform the expressions into terms with small constants.
Therefore, an alternative approach is to substitute t=x-99, which transforms the equation into t(t-2)=8 \\rightarrow t^2-2t-8=0. This can be easily factorized into (t-4)(t+2)=0. Hence, t=4 or t=-2, leading to the same results x=97 or x=103. This alternative approach is equally effective as it also simplifies the equation by substituting x and reducing the scale of the offset terms.

Follow-up Reasoning:
Extending the idea of substitution, consider the equation x(x+1)(x+2)(x+3)=360. We notice that x(x+3)=x^2+3x, and (x+1)(x+2)=x^2+3x+2. Therefore, to simplify the expression, we set the common term x^2+3x as t, which transforms the equation into t(t+2)=360 \\rightarrow t^2+2t-360=0 \\rightarrow t=-20 or t=18. If t=-20, then x^2+3x+20=0. Here, the discriminant Δ=-71<0, resulting in no real solutions for x. If t=18, then x^2+3x-18=0, so x=3 or x=-6. This scenario reiterates the importance of identifying common components of x to streamline the equation through substitution.

Now write a reflection section for the following case based on the examples above. Make sure to use "Alternative Reasoning:" and "Follow-up Reasoning:" to separate the two components.

Original Problem:
$$QUESTION$$

Solution to the Original Problem:
$$RESPONSE$$

Alternative Reasoning:
"""
