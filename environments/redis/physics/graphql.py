import graphene


class ControlCar(graphene.Mutation):
    """Control car mutation"""
    class Arguments:
        right = graphene.Float()
        throttle = graphene.Float()
        reset = graphene.Boolean()

    car = graphene.Field(lambda: Car)

    def mutate(self, rotation, throttle):
        car = Car(rotation=rotation, throttle=throttle, reset=False)
        return ControlCar(car=car)


class Car(graphene.ObjectType):
    """Represents the car"""
    rotation = graphene.Float()
    throttle = graphene.Float()


class MyMutations(graphene.ObjectType):
    control_car = ControlCar.Field()


# We must define a query for our schema
class Query(graphene.ObjectType):
    car = graphene.Field(Car)


# Define the schema
schema = graphene.Schema(query=Query, mutation=MyMutations)





def control_car(rotation,throttle,reset=False):
    """Control the Car"""
    query = """
        mutation Car(rotation: Float, throttle: $throttle $reset: Boolean) {
          controlCar(right: $rotation, throttle: $throttle, reset: $reset) {
            car {
              rotation
              throttle
            }
          }
        }
    """

    #print('Sending control signal',(rotation,throttle))
    return schema.execute(query, context_value={
        'rotation': rotation,
        'throttle': throttle,
        'reset': reset
    })




