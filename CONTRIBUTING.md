# Contributing

Thank you for thinking to contribute! (or just being curious about this page :smiley:)

The bigger a project gets, the more necessary to have rules to focus everybody's efforts in an efficient way.
This page gives some guidelines that should be followed to contribute to the project. Obviously, these rules will evolve with time and you are very welcome to suggest modifications of them if you believe they do not make sense.

Please always keep in mind that SciFin serves a certain purpose, and as such its development should make the code move towards that goal. You can read about the objectives of SciFin in the [Readme](https://github.com/SciFin-Team/SciFin/blob/master/README.md) file.


## Reporting bugs

...


## Suggesting ideas

If you think some idea or feature could fit well within SciFin, please do not hesitate to send us an email.



## Branches, commits and tests

Nobody likes to create branches, merge them, resolve conflicts, etc. So why should we bother? Well, that question has the same answer as why we should eat these non-exciting vegetables like broccoli, cabbage, and spinach... because they are very healthy for us! :smiley: If we don't create branches, one can be almost certain that when the code is too large to check that everything works fine after we modified some function, the modifications that we thought would not affect anyone else will actually disrupt other people's work. **It is thus extremely important to create local development branches and merge them with the master branch only when we are sure that disruption is minimized.**

There would be no branches without commits, so let's say a few words about them. It is advised to do a commit of your work each time you finish some task that do not involve too many modifications (50 lines of code is a good estimate). **Committing too little or too much both makes a commit useless**, so keep an eye on your commits size. as for the description of your commit, it is advised to tell which function you worked on, in which class or package, and give a short explanation of what you did. Keep in mind that a commit is a message in a bottle for someone who does not know (at all) what you are doing, or for your own self in a not-so-distant future (who may not know at all what you were doing!).

If dealing with branches is as appealing as eating broccolis, creating unit tests is probably as boring as washing the dishes! But that also is a necessity! **Every function that can have a unit test should see a unit test being written.** The best moment to write that test is right after writing the function, preferably in a new commit. One should keep in mind that unit testing is like a safeguard and a guarantee of solidity for the future code, hence big saving of time and energy.



## Coding conventions

**Generalities:**

- Every function or class we write should have at least a docstring with a short description of its purpose. Doctrings start with """ and end with """.

- Every function or class whose internal operations are not obvious to a "naive reader" should have comments explaining the main operations leading to the result. Comments start with #.

- When a function has several arguments (or when a class has several attributes), it is advised to complete the docstring with the list of arguments (or attributes). The more the arguments (or attributes), the stronger the recommendation to implement that documentation.

- When returned objects are not obvious, a Returns section should be added. A Notes section can also be used to give precisions about the choices made for development and to give references to the reader.


**Templates:**

To help development and avoid too much typing, some templates for functions and classes are provided in [coding_templates.md](https://github.com/SciFin-Team/SciFin/wiki/docs/coding_templates.md).


**Syntax advice:**

Going more into details, some good practice to keep the code clear are the following:

- When deciding the name of a function, try to follow the pattern method_quantity, for example "historical_variance". Functions should not have capital letters, but classes should (like "TimeSeries").
- For arguments of a function, place most important first and least important last so that from left to write we go from most important to least important arguments.
- Avoid starting description of a function with "Function that ..." or "Method that ...". No need for that.
- Avoid starting description of parameters with "The" or "A".
- Use 'DataFrame' to refer to the pandas object and 'data frame' to refer to it in a more general context. Same for classes like 'TimeSeries' with 'time series' or 'CatTimeSeries' with 'categorical time series'.
- Try to use the base form of verbs in your comments, e.g. 'Create a variable' instead of 'Creates a variable' or (worth) 'Creating a variable'. Imagine you are writing a recipe.
- Cite only reliable sources, preferentially in the "Notes" sections.
- Keep small caps for variables, except for a one-letter variable name where it matters less (e.g. N, T).


## Submitting changes


...

