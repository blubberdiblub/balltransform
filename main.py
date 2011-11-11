#!/usr/bin/python2.7

import collections, numpy, pygame

def quaternion_about_axis(angle, axis):
    q = numpy.array((0.0, axis[0], axis[1], axis[2]), dtype=numpy.float)
    q[1:4] *= numpy.sin(angle/2.0) / numpy.dot(q[1:4], q[1:4])
    q[0] = numpy.cos(angle/2.0)
    return q

def quaternion_matrix(quaternion):
    q = numpy.array(quaternion, dtype=numpy.float, copy=True)
    n = numpy.dot(q, q)
    if n == 0.0:
        return numpy.identity(4)
    q *= numpy.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0),
        (    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0),
        (    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)), dtype=numpy.float)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return numpy.array((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0), dtype=numpy.float)

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = numpy.dot(q0, q1)
    if d == 1.0 or d == -1.0:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        numpy.negative(q1, q1)
    angle = numpy.arccos(d) + spin * numpy.pi
    if angle == 0.0:
        return q0
    isin = 1.0 / numpy.sin(angle)
    q0 *= numpy.sin((1.0 - fraction) * angle) * isin
    q1 *= numpy.sin(fraction * angle) * isin
    q0 += q1
    return q0

def random_quaternion():
    rand = numpy.random.rand(3)
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = numpy.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array((numpy.cos(t2)*r2, numpy.sin(t1)*r1,
                        numpy.cos(t1)*r1, numpy.sin(t2)*r2), dtype=numpy.float)

def unit_vector(data):
    data = numpy.array(data, dtype=numpy.float, copy=True)
    data /= numpy.sqrt(numpy.dot(data, data))
    return data

pygame.init()

background_color = pygame.Color('black')
foreground_color = pygame.Color('white')
screen = pygame.display.set_mode((640, 480))
font = pygame.font.SysFont('fixed', 16)
text = font.render('Bitte warten!', True, foreground_color)
screen.blit(text, (screen.get_rect().centerx - text.get_rect().centerx, screen.get_rect().centery - text.get_rect().centery))
del text
pygame.display.flip()

original_sphere = pygame.image.load('sphere.png')
spheres = [None]*201
for i in range(0, 201):
    spheres[i] = pygame.transform.smoothscale(original_sphere, (i * 2 + 1, i * 2 + 1))
del original_sphere

animation_sequence = [
{'static': 7.0, 'fading': 1.0, 'balls': (
    (numpy.array([-3.0, -3.0, -3.0]), 1.0),
    (numpy.array([ 0.0, -3.0, -3.0]), 1.0),
    (numpy.array([ 3.0, -3.0, -3.0]), 1.0),
    (numpy.array([-3.0,  0.0, -3.0]), 1.0),
    (numpy.array([ 0.0,  0.0, -3.0]), 1.0),
    (numpy.array([ 3.0,  0.0, -3.0]), 1.0),
    (numpy.array([-3.0,  3.0, -3.0]), 1.0),
    (numpy.array([ 0.0,  3.0, -3.0]), 1.0),
    (numpy.array([ 3.0,  3.0, -3.0]), 1.0),
    (numpy.array([-3.0, -3.0,  0.0]), 1.0),
    (numpy.array([ 0.0, -3.0,  0.0]), 1.0),
    (numpy.array([ 3.0, -3.0,  0.0]), 1.0),
    (numpy.array([-3.0,  0.0,  0.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  0.0]), 1.0),
    (numpy.array([ 3.0,  0.0,  0.0]), 1.0),
    (numpy.array([-3.0,  3.0,  0.0]), 1.0),
    (numpy.array([ 0.0,  3.0,  0.0]), 1.0),
    (numpy.array([ 3.0,  3.0,  0.0]), 1.0),
    (numpy.array([-3.0, -3.0,  3.0]), 1.0),
    (numpy.array([ 0.0, -3.0,  3.0]), 1.0),
    (numpy.array([ 3.0, -3.0,  3.0]), 1.0),
    (numpy.array([-3.0,  0.0,  3.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  3.0]), 1.0),
    (numpy.array([ 3.0,  0.0,  3.0]), 1.0),
    (numpy.array([-3.0,  3.0,  3.0]), 1.0),
    (numpy.array([ 0.0,  3.0,  3.0]), 1.0),
    (numpy.array([ 3.0,  3.0,  3.0]), 1.0),
    )},
{'static': 7.0, 'fading': 1.0, 'balls': (
    (numpy.array([ 0.0, -6.4,  0.0]), 1.0),
    (numpy.array([-1.45,-5.0,  0.0]), 1.0),
    (numpy.array([-3.3, -4.2,  0.0]), 1.0),
    (numpy.array([-5.0, -3.2,  0.0]), 1.0),
    (numpy.array([-6.3, -1.7,  0.0]), 1.0),
    (numpy.array([-7.0,  0.2,  0.0]), 1.0),
    (numpy.array([-7.0,  2.2,  0.0]), 1.0),
    (numpy.array([-6.4,  4.1,  0.0]), 1.0),
    (numpy.array([-5.1,  5.6,  0.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  0.0]), 0.0),
    (numpy.array([-3.15, 6.0,  0.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  0.0]), 0.0),
    (numpy.array([-1.2,  5.6,  0.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  0.0]), 0.0), #
    (numpy.array([ 0.0,  4.0,  0.0]), 1.0), #
    (numpy.array([ 0.0,  0.0,  0.0]), 0.0),
    (numpy.array([ 1.2,  5.6,  0.0]), 1.0),
    (numpy.array([ 0.0,  0.0,  0.0]), 0.0),
    (numpy.array([ 3.15, 6.0,  0.0]), 1.0), #
    (numpy.array([ 5.1,  5.6,  0.0]), 1.0),
    (numpy.array([ 6.4,  4.1,  0.0]), 1.0),
    (numpy.array([ 7.0,  2.2,  0.0]), 1.0),
    (numpy.array([ 7.0,  0.2,  0.0]), 1.0), #
    (numpy.array([ 6.3, -1.7,  0.0]), 1.0),
    (numpy.array([ 5.0, -3.2,  0.0]), 1.0),
    (numpy.array([ 3.3, -4.2,  0.0]), 1.0),
    (numpy.array([ 1.45,-5.0,  0.0]), 1.0),
    )},
]

orientation = numpy.array((1.0, 0.0, 0.0, 0.0), dtype=numpy.float)
rotation = quaternion_about_axis(numpy.radians(1.0), (0.0, 0.3, 1.0))
#rotation = orientation.copy()

animation_index = 0
relative_time = 0.0
fadestart = animation_sequence[0]['static']
duration = fadestart + animation_sequence[0]['fading']
primary_balls = animation_sequence[0]['balls']
secondary_balls = animation_sequence[1 % len(animation_sequence)]['balls']
balls = [[numpy.array((0.0, 0.0, 0.0)), 0.0] for ball in primary_balls]
cumulative_time = pygame.time.get_ticks() / 1000.0

clock = pygame.time.Clock()
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
    relative_time = pygame.time.get_ticks() / 1000.0 - cumulative_time
    while relative_time >= duration:
        cumulative_time = cumulative_time + duration
        relative_time = relative_time - duration
        animation_index = (animation_index + 1) % len(animation_sequence)
        fadestart = animation_sequence[animation_index]['static']
        duration = fadestart + animation_sequence[animation_index]['fading']
        primary_balls = animation_sequence[animation_index]['balls']
        secondary_balls = animation_sequence[(animation_index + 1) % len(animation_sequence)]['balls']
    orientation = quaternion_multiply(orientation, rotation)
    orientation_matrix = quaternion_matrix(orientation)    
    if relative_time <= fadestart:
        for ball, primary in zip(balls, primary_balls):
            v = primary[0]
            if isinstance(v, collections.Callable):
                v = v(relative_time)
            if v is not None:
                ball[0] = numpy.dot(v, orientation_matrix[:3,:3].T)
            r = primary[1]
            if isinstance(r, collections.Callable):
                r = r(relative_time)
            ball[1] = r
    else:
        for ball, primary, secondary in zip(balls, primary_balls, secondary_balls):
            v1 = primary[0]
            if isinstance(v1, collections.Callable):
                v1 = v1(relative_time)
            v2 = secondary[0]
            if isinstance(v2, collections.Callable):
                v2 = v2(relative_time - duration)
            if v1 is not None and v2 is not None:
                ball[0] = numpy.dot((numpy.array(v1) * (duration - relative_time) +
                                     numpy.array(v2) * (relative_time - fadestart)) / (duration - fadestart), orientation_matrix[:3,:3].T)
            r1 = primary[1]
            if isinstance(r1, collections.Callable):
                r1 = r1(relative_time)
            r2 = secondary[1]
            if isinstance(r2, collections.Callable):
                r2 = r2(relative_time)
            ball[1] = (r1 * (duration - relative_time) +
                       r2 * (relative_time - fadestart)) / (duration - fadestart)
    balls.sort(key=lambda ball: ball[0][2], reverse=True)
    screen.fill(background_color)
    for ball in balls:
        v = ball[0]
        z = v[2] + 10.0
        if z > 0.0:
            z = 200.0 / z
            r = ball[1] * z
            if r > 0.0:
                r = int(r)
                if r < len(spheres):
                    s = spheres[r]
                    r = s.get_rect()
                    r.center = screen.get_rect().center
                    r.move_ip(v[0] * z, -v[1] * z)
                    screen.blit(s, r)
    pygame.display.flip()

pygame.display.quit()
    