using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.Areas.Admin.Controllers
{
    [Authorize(Roles = "Admin")]
    public class trainee_coursesController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();
        // GET: Admin/trainee_courses
        public ActionResult Index(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var trainees=db.Trainee_Courses.Where(t=>t.Course_id == id);
            return View(trainees);
        }
    }
}