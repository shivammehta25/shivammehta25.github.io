---
layout: post
title: Complex Numbers With Python Object Oriented Approach
date: 2019-03-26 13:54 +0100
categories: [programming]
tags: [python, object-orignted-programming, complex-number]
image:
  path: thumbnail.jpeg
  alt: Complex Numbers With Python Object Oriented Approach
---
## (Migrated from old blog)

Alrighty, Let's Begin with something very easy and this will be the only easy task I think I have encountered here. Let's begin with first programming task, writing Complex Number with the help of classes, overriding the addition and multiplication and other mathematical methods.
Lets begin with Complex Number so Complex number has two parts: real number and complex number lets create a class...

Let's begin with first programming task, writing Complex Number with the help of classes, overriding the addition and multiplication and other mathematical methods.
Let's begin with Complex Number so Complex number has two parts: real number and complex number lets create a class...

```python
class ComplexNumber:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
```

So with this, you can initialize a Complex Number with real and imaginary values. Lets, add try to create an object.

```python
a = ComplexNumber(2,5)
```

So now the Complex Number has real value as 2 and 5 as imaginary part lets override the basic methods of `__add__` , `__sub__` , `__str__`, `__mul__` and `__eq__` .
As the name suggests `__add__` will add two complex numbers, `__sub__` will subtract two numbers, `__str__` will print a stringified version of the number and `__mul__` will multiply it and `__eq__` will check for equality if they are equal or not.

```python
class ComplexNumber:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imaginary + other.imaginary)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real, self.imaginary - other.imaginary)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real, self.imaginary * other.imaginary)

    def __eq__(self, other):
        return self.real == other.real and self.imaginary == other.imaginary

    def __str__(self):
        return '{0} + {1}i'.format(self.real, self.imaginary) if self.real else '{0}i'.format(self.imaginary if self.imaginary else '')
```

Testing it with main function lets see how it runs..

```python
if __name__ == '__main__':
    a = ComplexNumber(0, 9)
    b = ComplexNumber(10, 5)
    c = ComplexNumber(5, 2)
    print('Str: a = {0} , b = {1} , c = {2} '.format(a, b, c))
    print('Addition: {0}'.format(a+b))
    print('Subtraction: {0}'.format(b-a))
    print('Multiplication: {0}'.format(a*b))
    print('Equality For True: {0}'.format(a == c))
    print('Equality for False: {0}'.format(a == b))
```

Output:

```text
#We Get the Output As :
Str: a = 9i , b = 10 + 5i , c = 5 + 2i
Addition: 10 + 14i
Subtraction: 10 + -4i
Multiplication: 45i
Equality For True: False
Equality for False: False
```

Voila our first blog post. Lets keep them coming ;)
